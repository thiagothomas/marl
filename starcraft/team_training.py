from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ml.ppo import PPOAgent

from .env import StarCraftTeamEnv
from .map_io import load_grid
from .runtime import RuntimeParameters, derive_runtime_parameters, load_runtime_from_metadata
from .scenario import Scenario
from .training import compute_rollout_schedule


Coordinate = Tuple[int, int]


@dataclass(frozen=True)
class TeamAgentScenario:
    scenario: Scenario
    runtime: RuntimeParameters
    config_path: Path


@dataclass(frozen=True)
class TeamFormation:
    team_name: str
    map_id: str
    map_path: Path
    grid_width: int
    grid_height: int
    agents: List[TeamAgentScenario]

    @property
    def scenario_ids(self) -> List[str]:
        return [agent.scenario.scenario_id for agent in self.agents]


@dataclass
class TeamTrainingConfig:
    formation: TeamFormation
    episodes: int
    models_dir: Path
    device: str = "cpu"
    entropy_coef: float = 0.003
    entropy_coef_final: float = 0.001
    entropy_anneal_fraction: float = 0.8
    rollout_length: Optional[int] = None
    num_envs: Optional[int] = None
    total_batch_size: Optional[int] = None
    min_rollout_length: int = 0
    minibatch_size: Optional[int] = None
    ppo_epochs: int = 4
    reward_mode: str = "sum"
    step_penalty_scale: float = 1.5
    progress_scale_multiplier: float = 0.5


def load_team_formation(
    team_dir: Path,
    *,
    repo_root: Optional[Path] = None,
) -> TeamFormation:
    """Load a team formation definition from saved single-agent policies."""
    if not team_dir.exists():
        raise FileNotFoundError(f"Team directory not found: {team_dir}")

    scenario_entries: List[Tuple[str, Path]] = []
    for scenario_dir in sorted(path for path in team_dir.iterdir() if path.is_dir()):
        config_path = scenario_dir / "PPOAgent" / "config.json"
        if config_path.exists():
            scenario_entries.append((scenario_dir.name, config_path))

    if not scenario_entries:
        raise FileNotFoundError(f"No scenario configs found under {team_dir}")

    root = repo_root or Path.cwd()
    agents: List[TeamAgentScenario] = []
    resolved_map_path: Optional[Path] = None
    resolved_map_id: Optional[str] = None
    grid_width = 0
    grid_height = 0

    for scenario_name, config_path in scenario_entries:
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse {config_path}: {err}") from err

        map_id = str(raw.get("map_id") or "").strip()
        map_path_raw = raw.get("map_path")
        if not map_id or not map_path_raw:
            raise ValueError(f"{config_path} missing map_id or map_path")

        map_path = _resolve_map_path(map_path_raw, root)
        if resolved_map_path is None:
            resolved_map_path = map_path
            resolved_map_id = map_id
            grid = load_grid(map_path)
            grid_height, grid_width = grid.shape
        else:
            if map_path != resolved_map_path:
                raise ValueError(
                    f"Team mixes multiple map files ({resolved_map_path} vs {map_path})"
                )
            if map_id != resolved_map_id:
                raise ValueError(
                    f"Team mixes multiple map ids ({resolved_map_id} vs {map_id})"
                )

        start = _parse_coordinate(raw.get("start"), "start", config_path)
        goal = _parse_coordinate(raw.get("goal"), "goal", config_path)
        bucket = int(raw.get("bucket", 0))
        optimal_length = float(raw.get("optimal_length", 0.0))
        scenario_id = str(raw.get("scenario_id") or scenario_name)
        scenario_index = _parse_scenario_index(scenario_id)

        scenario = Scenario(
            map_id=map_id,
            map_path=map_path,
            bucket=bucket,
            width=grid_width,
            height=grid_height,
            start=start,
            goal=goal,
            optimal_length=optimal_length,
            index=scenario_index,
        )

        runtime = load_runtime_from_metadata(config_path, scenario)
        if runtime is None:
            allow_diagonal = bool(raw.get("runtime", {}).get("allow_diagonal_actions", True))
            view_mode = str(raw.get("runtime", {}).get("view_mode", "moore"))
            view_radius = int(raw.get("runtime", {}).get("view_radius", 1))
            runtime = derive_runtime_parameters(
                scenario,
                None,
                allow_diagonal_actions=allow_diagonal,
                view_mode=view_mode,
                view_radius=view_radius,
            )

        agents.append(
            TeamAgentScenario(
                scenario=scenario,
                runtime=runtime,
                config_path=config_path,
            )
        )

    if resolved_map_path is None or resolved_map_id is None:
        raise RuntimeError("Unable to resolve map information for team formation")

    return TeamFormation(
        team_name=team_dir.name,
        map_id=resolved_map_id,
        map_path=resolved_map_path,
        grid_width=grid_width,
        grid_height=grid_height,
        agents=agents,
    )


def train_team_formation(config: TeamTrainingConfig):
    """Train a PPO policy that controls all agents in a team formation jointly."""
    reward_mode = config.reward_mode.strip().lower()
    if reward_mode not in {"sum", "mean"}:
        raise ValueError("reward_mode must be either 'sum' or 'mean'")

    grid = load_grid(config.formation.map_path)
    scenarios = [agent.scenario for agent in config.formation.agents]
    runtimes = [agent.runtime for agent in config.formation.agents]
    scaled_runtimes = [_scale_runtime(rt, config) for rt in runtimes]

    def env_factory():
        return StarCraftTeamEnv(
            grid=grid,
            scenarios=scenarios,
            runtimes=scaled_runtimes,
            team_name=config.formation.team_name,
            reward_mode=reward_mode,
        )

    aggregate_runtime = _aggregate_runtime(scaled_runtimes)
    rollout_length, total_batch, resolved_envs = compute_rollout_schedule(aggregate_runtime, config)
    if config.minibatch_size is None:
        minibatch_size = min(256, total_batch)
    else:
        minibatch_size = min(max(1, config.minibatch_size), total_batch)

    model_root = config.models_dir / config.formation.team_name
    goal_label = f"{config.formation.map_id}_team_{config.formation.team_name}"
    agent = PPOAgent(
        env_name=env_factory,
        models_dir=str(model_root),
        goal_hypothesis=goal_label,
        episodes=config.episodes,
        device=config.device,
        entropy_coef=config.entropy_coef,
        entropy_coef_final=config.entropy_coef_final,
        entropy_anneal_fraction=config.entropy_anneal_fraction,
        rollout_length=rollout_length,
        num_envs=resolved_envs,
        batch_size=minibatch_size,
        epochs=config.ppo_epochs,
    )

    # Persist metadata immediately so tooling can introspect in-progress runs.
    metadata_path = _write_team_metadata(
        agent.model_path,
        config,
        aggregate_runtime,
        scaled_runtimes,
        rollout_length,
        total_batch,
        minibatch_size,
        resolved_envs,
    )

    completed = agent.learn()

    metadata_path = _write_team_metadata(
        agent.model_path,
        config,
        aggregate_runtime,
        scaled_runtimes,
        rollout_length,
        total_batch,
        minibatch_size,
        resolved_envs,
    )
    return metadata_path, completed


# Helpers -----------------------------------------------------------------
def _resolve_map_path(map_path: str, repo_root: Path) -> Path:
    candidate = Path(map_path)
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    return candidate


def _scale_runtime(runtime: RuntimeParameters, config: TeamTrainingConfig) -> RuntimeParameters:
    step_penalty = runtime.step_penalty * max(1.0, float(config.step_penalty_scale))
    invalid_penalty = runtime.invalid_penalty * max(1.0, float(config.step_penalty_scale))
    progress_scale = runtime.progress_scale * max(0.0, float(config.progress_scale_multiplier))
    return RuntimeParameters(
        max_steps=runtime.max_steps,
        step_penalty=step_penalty,
        invalid_penalty=invalid_penalty,
        goal_reward=runtime.goal_reward,
        scale_used=runtime.scale_used,
        progress_scale=progress_scale,
        allow_diagonal_actions=runtime.allow_diagonal_actions,
        view_mode=runtime.view_mode,
        view_radius=runtime.view_radius,
    )


def _parse_coordinate(raw: Optional[Dict[str, object]], label: str, source: Path) -> Coordinate:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} coordinate missing or invalid in {source}")
    try:
        return int(raw["x"]), int(raw["y"])
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"{label} coordinate invalid in {source}: {err}") from err


def _parse_scenario_index(scenario_id: str) -> int:
    if "_line_" in scenario_id:
        suffix = scenario_id.rsplit("_line_", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    digits = "".join(ch for ch in scenario_id if ch.isdigit())
    return int(digits) if digits else 0


def _aggregate_runtime(runtimes: Sequence[RuntimeParameters]) -> RuntimeParameters:
    if not runtimes:
        raise ValueError("At least one runtime parameter set is required")
    base = runtimes[0]
    max_steps = max(runtime.max_steps for runtime in runtimes)
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


def _runtime_to_dict(runtime: RuntimeParameters) -> Dict[str, object]:
    return {
        "max_steps": runtime.max_steps,
        "step_penalty": runtime.step_penalty,
        "invalid_penalty": runtime.invalid_penalty,
        "goal_reward": runtime.goal_reward,
        "scale_used": runtime.scale_used,
        "progress_scale": runtime.progress_scale,
        "allow_diagonal_actions": runtime.allow_diagonal_actions,
        "view_mode": runtime.view_mode,
        "view_radius": runtime.view_radius,
    }


def _write_team_metadata(
    model_path: str,
    config: TeamTrainingConfig,
    aggregate_runtime: RuntimeParameters,
    agent_runtimes: Sequence[RuntimeParameters],
    rollout_length: int,
    total_batch: int,
    minibatch_size: int,
    num_envs: int,
) -> Path:
    target = Path(model_path) / "config.json"
    metadata = {
        "team_name": config.formation.team_name,
        "map_id": config.formation.map_id,
        "map_path": str(config.formation.map_path),
        "episodes": config.episodes,
        "num_agents": len(config.formation.agents),
        "reward_mode": config.reward_mode,
        "scenario_ids": config.formation.scenario_ids,
        "agents": [
            {
                "scenario_id": agent.scenario.scenario_id,
                "start": {"x": agent.scenario.start[0], "y": agent.scenario.start[1]},
                "goal": {"x": agent.scenario.goal[0], "y": agent.scenario.goal[1]},
                "bucket": agent.scenario.bucket,
                "optimal_length": agent.scenario.optimal_length,
                "runtime": _runtime_to_dict(runtime),
            }
            for agent, runtime in zip(config.formation.agents, agent_runtimes)
        ],
        "aggregate_runtime": _runtime_to_dict(aggregate_runtime),
        "training": {
            "entropy_coef": config.entropy_coef,
            "entropy_coef_final": config.entropy_coef_final,
            "entropy_anneal_fraction": config.entropy_anneal_fraction,
            "rollout_length": rollout_length,
            "target_rollout_length": config.rollout_length,
            "min_rollout_length": config.min_rollout_length,
            "total_batch_size": total_batch,
            "target_total_batch_size": config.total_batch_size,
            "num_envs": num_envs,
            "requested_num_envs": config.num_envs,
            "ppo_epochs": config.ppo_epochs,
            "minibatch_size": minibatch_size,
            "requested_minibatch_size": config.minibatch_size,
            "step_penalty_scale": config.step_penalty_scale,
            "progress_scale_multiplier": config.progress_scale_multiplier,
        },
    }
    target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target
