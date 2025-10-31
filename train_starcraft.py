#!/usr/bin/env python3
"""Train PPO policies for Moving AI StarCraft map scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

from starcraft import (
    TrainingConfig,
    compute_rollout_schedule,
    derive_runtime_parameters,
    load_scenarios,
    train_map_scenarios,
)


DEFAULT_MAPS_ROOT = Path("starcraft-maps")
DEFAULT_MODELS_DIR = Path("models/starcraft")
DEFAULT_DEVICE = "cpu"
DEFAULT_ENTROPY_COEF = 3e-3
DEFAULT_ENTROPY_COEF_FINAL = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO policies for StarCraft map scenarios.")
    parser.add_argument("--map-id", required=True, help="Map identifier (e.g., Aftershock).")
    parser.add_argument(
        "--scen-file",
        type=Path,
        default=None,
        help="Optional override for the scenario file path. If omitted, defaults to starcraft-maps/sc1-scen/<map>.map.scen.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Training episodes for the selected scenario.",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Zero-based index of the scenario to train.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=DEFAULT_ENTROPY_COEF,
        help="Initial entropy bonus. Higher values encourage exploration.",
    )
    parser.add_argument(
        "--entropy-coef-final",
        type=float,
        default=DEFAULT_ENTROPY_COEF_FINAL,
        help="Final entropy coefficient after annealing (set to 0 to disable).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=0,
        help="Number of parallel environments sampled per PPO update. Use 0 to auto-select.",
    )
    parser.add_argument(
        "--total-batch",
        type=int,
        default=0,
        help="Target total batch size per PPO update. Set to 0 to auto-select based on scenario.",
    )
    parser.add_argument(
        "--rollout-per-env",
        type=int,
        default=None,
        help="Override rollout steps collected per environment before each update.",
    )
    parser.add_argument(
        "--min-rollout",
        type=int,
        default=0,
        help="Minimum rollout length per environment after applying caps.",
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
        help="Minibatch size for PPO updates. Use 0 to auto-scale to the rollout size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maps_root = DEFAULT_MAPS_ROOT
    models_dir = DEFAULT_MODELS_DIR

    scen_path = args.scen_file
    if scen_path is None:
        scen_path = maps_root / "sc1-scen" / f"{args.map_id}.map.scen"

    if not scen_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scen_path}")

    all_scenarios = load_scenarios(scen_path)
    if not all_scenarios:
        raise ValueError(f"No scenarios found in {scen_path}")

    scenario_map_ids = {scenario.map_id for scenario in all_scenarios}
    if args.map_id not in scenario_map_ids:
        raise ValueError(
            f"Scenario file {scen_path} does not contain map '{args.map_id}'. "
            f"Available map ids: {sorted(scenario_map_ids)}"
        )

    if args.scenario_index < 0 or args.scenario_index >= len(all_scenarios):
        raise ValueError(
            f"Scenario index {args.scenario_index} out of range (0-{len(all_scenarios)-1})."
        )
    scenario = all_scenarios[args.scenario_index]
    if scenario.map_id != args.map_id:
        raise ValueError(
            f"Scenario at index {args.scenario_index} in {scen_path} is for map "
            f"'{scenario.map_id}', but --map-id requested '{args.map_id}'."
        )
    runtime_preview = derive_runtime_parameters(
        scenario,
        None,
        allow_diagonal_actions=True,
        view_mode="moore",
        view_radius=1,
    )

    requested_num_envs = args.num_envs if args.num_envs > 0 else None
    total_batch_target = args.total_batch if args.total_batch > 0 else None

    config = TrainingConfig(
        map_id=args.map_id,
        scenarios=[scenario],
        episodes=args.episodes,
        models_dir=models_dir,
        device=DEFAULT_DEVICE,
        entropy_coef=args.entropy_coef,
        entropy_coef_final=args.entropy_coef_final,
        allow_diagonal_actions=True,
        view_mode="moore",
        view_radius=1,
        num_envs=None if requested_num_envs is None else max(1, requested_num_envs),
        total_batch_size=total_batch_target,
        rollout_length=args.rollout_per_env,
        min_rollout_length=max(0, args.min_rollout),
        ppo_epochs=max(1, args.ppo_epochs),
        minibatch_size=None if args.minibatch_size <= 0 else args.minibatch_size,
    )

    rollout_preview, batch_preview, env_preview = compute_rollout_schedule(runtime_preview, config)
    if config.minibatch_size is None:
        minibatch_preview = min(256, batch_preview)
    else:
        minibatch_preview = min(config.minibatch_size, batch_preview)

    print("=" * 70)
    print(f"Selected scenario: {scenario.scenario_id}")
    print(f"  start: {scenario.start}, goal: {scenario.goal}")
    print(f"  optimal path length: {scenario.optimal_length:.3f} tiles")
    print(f"  derived max steps: {runtime_preview.max_steps} (scale {runtime_preview.scale_used:.2f})")
    print(f"  entropy coefficient: {args.entropy_coef:.4g} → {args.entropy_coef_final:.4g}")
    if requested_num_envs is None:
        print(f"  vectorized envs    : auto → {env_preview}")
    else:
        print(f"  vectorized envs    : {requested_num_envs} (resolved {env_preview})")
    target_batch_info = total_batch_target if total_batch_target is not None else "auto"
    print(f"  rollout/env        : {rollout_preview}")
    print(f"  batch/update       : {batch_preview} (target {target_batch_info})")
    if config.minibatch_size is None:
        print(f"  minibatch size     : auto → {minibatch_preview}")
    else:
        print(f"  minibatch size     : {config.minibatch_size} (resolved {minibatch_preview})")
    print(f"  PPO epochs         : {config.ppo_epochs}")
    print("  actions            : 8-directional (diagonals enabled)")
    print("  view window        : 3x3 occupancy patch")
    print("=" * 70)

    metadata_paths, interrupted = train_map_scenarios(config)

    checkpoints_root = models_dir / args.map_id / f"episodes_{args.episodes}"
    print("=" * 70)
    if interrupted:
        print(
            f"Training interrupted after {len(metadata_paths)} scenario(s). "
            f"Partial checkpoints stored under {checkpoints_root}."
        )
    else:
        print(f"Trained {len(metadata_paths)} scenarios for map '{args.map_id}'.")
    print(f"Checkpoints stored under {checkpoints_root}.")
    print("=" * 70)
    for metadata_path in metadata_paths:
        print(f"- {scenario.scenario_id}")
        print(f"  metadata  : {metadata_path}")
        print(f"  model dir : {metadata_path.parent}")


if __name__ == "__main__":
    main()
