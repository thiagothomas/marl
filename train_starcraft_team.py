#!/usr/bin/env python3
"""Train PPO team policies that control multiple StarCraft agents jointly."""

from __future__ import annotations

import argparse
from pathlib import Path

from starcraft import (
    RuntimeParameters,
    TeamTrainingConfig,
    compute_rollout_schedule,
    load_team_formation,
    train_team_formation,
)

DEFAULT_TEAMS_DIR = Path("models/starcraft/teams")
DEFAULT_TEAM_MODELS_DIR = Path("models/starcraft/team_policies")
DEFAULT_DEVICE = "cpu"
DEFAULT_ENTROPY_COEF = 3e-3
DEFAULT_ENTROPY_COEF_FINAL = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO policy that controls all agents in a StarCraft team formation."
    )
    parser.add_argument("--team-name", required=True, help="Team directory name under --teams-dir.")
    parser.add_argument(
        "--teams-dir",
        type=Path,
        default=DEFAULT_TEAMS_DIR,
        help="Directory that contains single-agent team formations.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_TEAM_MODELS_DIR,
        help="Output directory for trained team policies.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Training episodes for the team policy.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=DEFAULT_ENTROPY_COEF,
        help="Initial entropy coefficient.",
    )
    parser.add_argument(
        "--entropy-coef-final",
        type=float,
        default=DEFAULT_ENTROPY_COEF_FINAL,
        help="Final entropy coefficient after annealing.",
    )
    parser.add_argument(
        "--entropy-coef-anneal-fraction",
        type=float,
        default=0.8,
        help="Fraction of PPO updates over which to anneal the entropy coefficient.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=0,
        help="Vectorized environments per PPO update (0 = auto).",
    )
    parser.add_argument(
        "--total-batch",
        type=int,
        default=0,
        help="Target batch size per PPO update (0 = auto).",
    )
    parser.add_argument(
        "--rollout-per-env",
        type=int,
        default=None,
        help="Override rollout length collected per environment.",
    )
    parser.add_argument(
        "--min-rollout",
        type=int,
        default=0,
        help="Minimum rollout length enforced after scheduling.",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
        help="Gradient epochs per PPO update.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=0,
        help="Minibatch size per PPO update (0 = auto).",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["sum", "mean"],
        default="sum",
        help="Aggregate per-agent rewards using a sum or per-step mean.",
    )
    parser.add_argument(
        "--step-penalty-scale",
        type=float,
        default=1.5,
        help="Multiplier applied to per-step penalties for team training.",
    )
    parser.add_argument(
        "--progress-scale-multiplier",
        type=float,
        default=0.5,
        help="Multiplier applied to progress rewards for each agent.",
    )
    return parser.parse_args()


def _aggregate_runtime(formation_runtime) -> RuntimeParameters:
    """Build a summary runtime for schedule previews."""
    base = formation_runtime[0]
    max_steps = max(runtime.max_steps for runtime in formation_runtime)
    return RuntimeParameters(
        max_steps=max_steps,
        step_penalty=base.step_penalty,
        invalid_penalty=base.invalid_penalty,
        goal_reward=base.goal_reward,
        scale_used=base.scale_used,
        progress_scale=base.progress_scale,
        allow_diagonal_actions=base.allow_diagonal_actions,
        view_mode=base.view_mode,
        view_radius=base.view_radius,
    )


def main() -> None:
    args = parse_args()
    teams_dir = Path(args.teams_dir)
    models_dir = Path(args.models_dir)

    formation = load_team_formation(teams_dir / args.team_name)
    requested_num_envs = args.num_envs if args.num_envs > 0 else None
    total_batch_target = args.total_batch if args.total_batch > 0 else None

    config = TeamTrainingConfig(
        formation=formation,
        episodes=int(args.episodes),
        models_dir=models_dir,
        device=DEFAULT_DEVICE,
        entropy_coef=float(args.entropy_coef),
        entropy_coef_final=float(args.entropy_coef_final),
        entropy_anneal_fraction=float(args.entropy_coef_anneal_fraction),
        rollout_length=args.rollout_per_env,
        num_envs=None if requested_num_envs is None else max(1, requested_num_envs),
        total_batch_size=total_batch_target,
        min_rollout_length=max(0, args.min_rollout),
        minibatch_size=None if args.minibatch_size <= 0 else args.minibatch_size,
        ppo_epochs=max(1, args.ppo_epochs),
        reward_mode=args.reward_mode,
        step_penalty_scale=max(0.1, float(args.step_penalty_scale)),
        progress_scale_multiplier=max(0.0, float(args.progress_scale_multiplier)),
    )

    team_runtime = _aggregate_runtime([agent.runtime for agent in formation.agents])
    rollout_preview, batch_preview, env_preview = compute_rollout_schedule(team_runtime, config)
    if config.minibatch_size is None:
        minibatch_preview = min(256, batch_preview)
    else:
        minibatch_preview = min(config.minibatch_size, batch_preview)

    print("=" * 70)
    print(f"Team: {formation.team_name}")
    print(f"Map : {formation.map_id} ({formation.map_path})")
    print(f"Agents ({len(formation.agents)}):")
    for agent in formation.agents:
        start = agent.scenario.start
        goal = agent.scenario.goal
        print(
            f"  - {agent.scenario.scenario_id}: "
            f"start=({start[0]}, {start[1]}) → goal=({goal[0]}, {goal[1]})"
        )
    print(f"Reward mode        : {args.reward_mode}")
    print(f"Step penalty scale : {config.step_penalty_scale:.3f}")
    print(f"Progress scale mul.: {config.progress_scale_multiplier:.3f}")
    if requested_num_envs is None:
        print(f"Vectorized envs    : auto → {env_preview}")
    else:
        print(f"Vectorized envs    : {requested_num_envs} (resolved {env_preview})")
    print(f"Rollout/env        : {rollout_preview}")
    target_batch_info = total_batch_target if total_batch_target is not None else "auto"
    print(f"Batch/update       : {batch_preview} (target {target_batch_info})")
    if config.minibatch_size is None:
        print(f"Minibatch size     : auto → {minibatch_preview}")
    else:
        print(f"Minibatch size     : {config.minibatch_size} (resolved {minibatch_preview})")
    print(f"PPO epochs         : {config.ppo_epochs}")
    print(f"Episodes           : {config.episodes}")
    print("=" * 70)

    metadata_path, completed = train_team_formation(config)
    checkpoints_root = metadata_path.parents[2]

    print("=" * 70)
    if completed:
        print(
            f"✓ Trained team '{formation.team_name}' "
            f"({len(formation.agents)} agents) on map '{formation.map_id}'."
        )
    else:
        print(
            f"Training interrupted. The latest checkpoint (if any) lives under {checkpoints_root}."
        )
    print(f"Metadata saved to  : {metadata_path}")
    print(f"Checkpoints folder : {checkpoints_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
