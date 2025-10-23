from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ml.ppo import PPOAgent

from .env import StarCraftScenarioEnv
from .map_io import load_grid
from .scenario import Scenario
from .runtime import RuntimeParameters, derive_runtime_parameters


@dataclass
class TrainingConfig:
    map_id: str
    scenarios: List[Scenario]
    episodes: int
    models_dir: Path
    device: str = "cpu"
    max_steps_scale: Optional[float] = None
    allow_diagonal_actions: bool = True
    view_mode: str = "moore"
    view_radius: int = 1
    fixed_step_penalty: Optional[float] = 0.01
    fixed_progress_scale: Optional[float] = 0.2
    entropy_coef: float = 0.003
    entropy_coef_final: float = 0.001
    entropy_anneal_fraction: float = 0.8
    rollout_length: Optional[int] = None
    num_envs: Optional[int] = None
    total_batch_size: Optional[int] = None
    min_rollout_length: int = 0
    minibatch_size: Optional[int] = None
    ppo_epochs: int = 4


def _suggest_rollout_length(max_steps: int) -> int:
    """Heuristic rollout length that grows with scenario horizon."""
    if max_steps <= 0:
        return 128
    rollout = 128
    while max_steps > rollout and rollout < 4096:
        rollout *= 2
    if max_steps > rollout:
        return 4096
    return rollout


def _auto_num_envs(max_steps: int) -> int:
    if max_steps <= 256:
        return 4
    if max_steps <= 1024:
        return 6
    if max_steps <= 4096:
        return 8
    return 8


def _auto_total_batch_size(max_steps: int) -> int:
    if max_steps <= 256:
        return 1024
    if max_steps <= 1024:
        return 2048
    if max_steps <= 4096:
        return 3072
    return 4096


def compute_rollout_schedule(
    runtime: RuntimeParameters,
    config: TrainingConfig
) -> Tuple[int, int, int]:
    """Resolve per-environment rollout length, total batch size, and num envs."""
    num_envs = config.num_envs or _auto_num_envs(runtime.max_steps)
    num_envs = max(1, int(num_envs))

    target_batch = (
        config.total_batch_size
        if config.total_batch_size is not None
        else _auto_total_batch_size(runtime.max_steps)
    )

    base_rollout = config.rollout_length or _suggest_rollout_length(runtime.max_steps)
    per_env_rollout = base_rollout

    if target_batch:
        target_per_env = math.ceil(target_batch / num_envs)
        per_env_rollout = min(per_env_rollout, target_per_env)

    per_env_rollout = max(1, int(per_env_rollout))
    if config.min_rollout_length > 0:
        per_env_rollout = max(per_env_rollout, int(config.min_rollout_length))

    per_env_rollout = min(per_env_rollout, int(runtime.max_steps))
    total_batch = per_env_rollout * num_envs
    return per_env_rollout, total_batch, num_envs


def train_map_scenarios(config: TrainingConfig) -> Tuple[List[Path], bool]:
    map_path = config.scenarios[0].map_path
    grid = load_grid(map_path)

    saved_paths: List[Path] = []
    interrupted = False
    for scenario in config.scenarios:
        runtime = derive_runtime_parameters(
            scenario,
            config.max_steps_scale,
            allow_diagonal_actions=config.allow_diagonal_actions,
            view_mode=config.view_mode,
            view_radius=config.view_radius,
            fixed_step_penalty=config.fixed_step_penalty,
            fixed_progress_scale=config.fixed_progress_scale,
        )

        def env_factory(scenario=scenario, runtime=runtime):
            return StarCraftScenarioEnv(
                grid=grid,
                scenario=scenario,
                max_steps=runtime.max_steps,
                step_penalty=runtime.step_penalty,
                invalid_penalty=runtime.invalid_penalty,
                goal_reward=runtime.goal_reward,
                progress_scale=runtime.progress_scale,
                allow_diagonal_actions=runtime.allow_diagonal_actions,
                view_mode=runtime.view_mode,
                view_radius=runtime.view_radius,
            )

        rollout_length, total_batch, resolved_envs = compute_rollout_schedule(runtime, config)
        if config.minibatch_size is None:
            minibatch_size = min(256, total_batch)
        else:
            minibatch_size = min(max(1, config.minibatch_size), total_batch)

        agent = PPOAgent(
            env_name=env_factory,
            models_dir=str(config.models_dir / config.map_id),
            goal_hypothesis=scenario.scenario_id,
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
        completed = agent.learn()
        metadata_path = _write_metadata(
            agent.model_path,
            scenario,
            config,
            runtime,
            rollout_length,
            total_batch,
            minibatch_size,
            resolved_envs,
        )
        saved_paths.append(metadata_path)
        if not completed:
            interrupted = True
            print("Interrupt received; stopping remaining scenarios.")
            break

    return saved_paths, interrupted


def _write_metadata(
    model_path: Union[Path, str],
    scenario: Scenario,
    config: TrainingConfig,
    runtime: RuntimeParameters,
    rollout_length: int,
    total_batch: int,
    minibatch_size: int,
    num_envs: int,
) -> Path:
    model_dir = Path(model_path)
    metadata = {
        "map_id": scenario.map_id,
        "map_path": str(scenario.map_path),
        "scenario_id": scenario.scenario_id,
        "start": {"x": scenario.start[0], "y": scenario.start[1]},
        "goal": {"x": scenario.goal[0], "y": scenario.goal[1]},
        "episodes": config.episodes,
        "bucket": scenario.bucket,
        "optimal_length": scenario.optimal_length,
        "max_steps_scale": config.max_steps_scale,
        "entropy_coef": config.entropy_coef,
        "entropy_coef_final": config.entropy_coef_final,
        "entropy_anneal_fraction": config.entropy_anneal_fraction,
        "rollout_length": rollout_length,
        "target_rollout_length": config.rollout_length,
        "num_envs": num_envs,
        "requested_num_envs": config.num_envs,
        "total_batch_size": total_batch,
        "target_total_batch_size": config.total_batch_size,
        "min_rollout_length": config.min_rollout_length,
        "ppo_epochs": config.ppo_epochs,
        "minibatch_size": minibatch_size,
        "requested_minibatch_size": config.minibatch_size,
        "runtime": {
            "scenario_id": scenario.scenario_id,
            "max_steps": runtime.max_steps,
            "step_penalty": runtime.step_penalty,
            "invalid_penalty": runtime.invalid_penalty,
            "goal_reward": runtime.goal_reward,
            "scale_used": runtime.scale_used,
            "progress_scale": runtime.progress_scale,
            "allow_diagonal_actions": runtime.allow_diagonal_actions,
            "view_mode": runtime.view_mode,
            "view_radius": runtime.view_radius,
        },
    }
    metadata_path = model_dir / "config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path
