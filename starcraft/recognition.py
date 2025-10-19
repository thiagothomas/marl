from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ml.ppo import PPOAgent

from .env import StarCraftScenarioEnv
from .map_io import load_grid
from .scenario import Scenario
from .runtime import derive_runtime_parameters, load_runtime_from_metadata, RuntimeParameters


@dataclass
class EvaluationConfig:
    map_id: str
    scenarios: List[Scenario]
    models_dir: Path
    train_episodes: int
    rollouts: int = 5
    device: str = "cpu"
    max_steps_scale: Optional[float] = None


def evaluate_map_scenarios(config: EvaluationConfig) -> List[Dict[str, float]]:
    grid = load_grid(config.scenarios[0].map_path)
    results: List[Dict[str, float]] = []

    for scenario in config.scenarios:
        runtime = _resolve_runtime_parameters(config, scenario)

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
            episodes=config.train_episodes,
            device=config.device,
        )
        agent.load_model()

        episode_rewards = []
        success_count = 0
        for _ in range(config.rollouts):
            env = env_factory()
            obs, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action_probs = agent.get_action_probabilities(obs)
                action = int(np.argmax(action_probs))
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1

            if terminated:
                success_count += 1
            episode_rewards.append(total_reward)

        results.append(
            {
                "scenario_id": scenario.scenario_id,
                "success_rate": success_count / config.rollouts,
                "mean_reward": float(np.mean(episode_rewards)),
                "std_reward": float(np.std(episode_rewards)),
                "max_steps": runtime.max_steps,
            }
        )

    return results


def _resolve_runtime_parameters(config: EvaluationConfig, scenario: Scenario) -> RuntimeParameters:
    policy_dir = (
        config.models_dir
        / config.map_id
        / f"episodes_{config.train_episodes}"
        / scenario.scenario_id
        / "PPOAgent"
    )
    metadata_runtime = load_runtime_from_metadata(policy_dir / "config.json", scenario)

    if config.max_steps_scale is not None:
        return derive_runtime_parameters(scenario, config.max_steps_scale)

    if metadata_runtime is not None:
        return metadata_runtime

    return derive_runtime_parameters(scenario, None)
