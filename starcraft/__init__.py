"""StarCraft map training and recognition helpers."""

from .env import StarCraftScenarioEnv
from .map_io import load_grid
from .recognition import EvaluationConfig, evaluate_map_scenarios
from .scenario import Scenario, load_scenarios
from .runtime import RuntimeParameters, derive_runtime_parameters, load_runtime_from_metadata
from .training import TrainingConfig, train_map_scenarios, compute_rollout_schedule

__all__ = [
    "StarCraftScenarioEnv",
    "load_grid",
    "Scenario",
    "load_scenarios",
    "TrainingConfig",
    "train_map_scenarios",
    "compute_rollout_schedule",
    "EvaluationConfig",
    "evaluate_map_scenarios",
    "RuntimeParameters",
    "derive_runtime_parameters",
    "load_runtime_from_metadata",
]
