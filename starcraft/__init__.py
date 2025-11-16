"""StarCraft map training and recognition helpers."""

from .env import StarCraftScenarioEnv, StarCraftTeamEnv
from .map_io import load_grid
from .recognition import EvaluationConfig, evaluate_map_scenarios
from .scenario import Scenario, load_scenarios
from .runtime import RuntimeParameters, derive_runtime_parameters, load_runtime_from_metadata
from .training import TrainingConfig, train_map_scenarios, compute_rollout_schedule
from .team_training import (
    TeamFormation,
    TeamTrainingConfig,
    load_team_formation,
    train_team_formation,
)

__all__ = [
    "StarCraftScenarioEnv",
    "StarCraftTeamEnv",
    "load_grid",
    "Scenario",
    "load_scenarios",
    "TrainingConfig",
    "train_map_scenarios",
    "compute_rollout_schedule",
    "TeamFormation",
    "TeamTrainingConfig",
    "load_team_formation",
    "train_team_formation",
    "EvaluationConfig",
    "evaluate_map_scenarios",
    "RuntimeParameters",
    "derive_runtime_parameters",
    "load_runtime_from_metadata",
]
