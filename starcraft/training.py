from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

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


def train_map_scenarios(config: TrainingConfig) -> List[Path]:
    map_path = config.scenarios[0].map_path
    grid = load_grid(map_path)

    saved_paths: List[Path] = []
    for scenario in config.scenarios:
        runtime = derive_runtime_parameters(scenario, config.max_steps_scale)

        def env_factory(scenario=scenario, runtime=runtime):
            return StarCraftScenarioEnv(
                grid=grid,
                scenario=scenario,
                max_steps=runtime.max_steps,
                step_penalty=runtime.step_penalty,
                invalid_penalty=runtime.invalid_penalty,
                goal_reward=runtime.goal_reward,
                progress_scale=runtime.progress_scale,
            )

        agent = PPOAgent(
            env_name=env_factory,
            models_dir=str(config.models_dir / config.map_id),
            goal_hypothesis=scenario.scenario_id,
            episodes=config.episodes,
            device=config.device,
        )
        agent.learn()
        metadata_path = _write_metadata(agent.model_path, scenario, config, runtime)
        saved_paths.append(metadata_path)

    return saved_paths


def _write_metadata(
    model_path: Union[Path, str],
    scenario: Scenario,
    config: TrainingConfig,
    runtime: RuntimeParameters,
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
        "runtime": {
            "scenario_id": scenario.scenario_id,
            "max_steps": runtime.max_steps,
            "step_penalty": runtime.step_penalty,
            "invalid_penalty": runtime.invalid_penalty,
            "goal_reward": runtime.goal_reward,
            "scale_used": runtime.scale_used,
            "progress_scale": runtime.progress_scale,
        },
    }
    metadata_path = model_dir / "config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path
