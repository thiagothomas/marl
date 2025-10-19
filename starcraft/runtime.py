from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .scenario import Scenario


@dataclass(frozen=True)
class RuntimeParameters:
    max_steps: int
    step_penalty: float
    invalid_penalty: float
    goal_reward: float
    scale_used: float
    progress_scale: float


def derive_runtime_parameters(
    scenario: Scenario,
    override_scale: Optional[float] = None,
    *,
    min_scale: float = 2.5,
    bucket_weight: float = 0.5,
    base_goal_reward: float = 1.0,
    invalid_penalty_multiplier: float = 4.0,
) -> RuntimeParameters:
    """
    Produce environment runtime parameters based on scenario metadata.

    Args:
        scenario: Scenario definition.
        override_scale: If provided, force max_steps to optimal_length * scale.
        min_scale: Lower bound for the automatically derived scale factor.
        bucket_weight: Multiplier applied to the scenario bucket when deriving scale.
        base_goal_reward: Reward applied upon reaching the goal.
        invalid_penalty_multiplier: Multiplier for the per-step penalty when
            the agent attempts to walk into a blocked tile.
    """
    scale = (
        override_scale
        if override_scale is not None
        else max(min_scale, 1.5 + bucket_weight * scenario.bucket)
    )

    max_steps = max(int(math.ceil(scenario.optimal_length * scale)), 1)
    step_penalty = 1.0 / max_steps
    invalid_penalty = invalid_penalty_multiplier * step_penalty

    # Encourage forward progress with a shaping term proportional to expected episode length.
    rough_episode_count = scenario.optimal_length or 1.0
    progress_scale = 4.0 * step_penalty * rough_episode_count
    progress_scale = float(max(0.05, min(1.0, progress_scale)))

    return RuntimeParameters(
        max_steps=max_steps,
        step_penalty=step_penalty,
        invalid_penalty=invalid_penalty,
        goal_reward=base_goal_reward,
        scale_used=scale,
        progress_scale=progress_scale,
    )


def load_runtime_from_metadata(
    metadata_path: Path,
    scenario: Scenario,
) -> Optional[RuntimeParameters]:
    if not metadata_path.exists():
        return None

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    runtime = data.get("runtime")
    if not runtime:
        return None

    # Sanity-check scenario id.
    stored = runtime.get("scenario_id") or data.get("scenario", {}).get("scenario_id")
    if stored and stored != scenario.scenario_id:
        return None

    try:
        return RuntimeParameters(
            max_steps=int(runtime["max_steps"]),
            step_penalty=float(runtime["step_penalty"]),
            invalid_penalty=float(runtime["invalid_penalty"]),
            goal_reward=float(runtime.get("goal_reward", 1.0)),
            scale_used=float(runtime.get("scale_used", 0.0)),
            progress_scale=float(runtime.get("progress_scale", 0.1)),
        )
    except (TypeError, ValueError, KeyError):
        return None
