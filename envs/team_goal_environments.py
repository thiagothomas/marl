"""Single-agent environments for training team-specific goal policies.

These environments are used to train individual agents for each team's goal.
During training, we train policies for:
- Team goals (e.g., reach top-right for Team A, reach top-left for Team B)

In the multi-agent recognition phase, we'll match observed behaviors
against these trained policies to determine team membership and goals.
"""

from .grid_world import GridWorldBase
import numpy as np

STEP_PENALTY = 0.01
DISTANCE_WEIGHT = 0.01


class TeamGoalTopRight(GridWorldBase):
    """Environment for training agents to reach top-right (Team A's goal)."""

    def _set_goal_position(self):
        """Set goal to top-right corner."""
        self.goal_pos = np.array([self.size - 1, 0], dtype=np.int32)

    def _calculate_reward(self) -> float:
        """Reward for reaching top-right corner."""
        if self._is_goal_reached():
            return 10.0  # Big reward for reaching goal
        distance = np.linalg.norm(self.agent_pos.astype(float) - self.goal_pos.astype(float))
        return -(STEP_PENALTY + DISTANCE_WEIGHT * distance)


class TeamGoalTopLeft(GridWorldBase):
    """Environment for training agents to reach top-left (Team B's goal)."""

    def _set_goal_position(self):
        """Set goal to top-left corner."""
        self.goal_pos = np.array([0, 0], dtype=np.int32)

    def _calculate_reward(self) -> float:
        """Reward for reaching top-left corner."""
        if self._is_goal_reached():
            return 10.0
        distance = np.linalg.norm(self.agent_pos.astype(float) - self.goal_pos.astype(float))
        return -(STEP_PENALTY + DISTANCE_WEIGHT * distance)


class TeamGoalBottomRight(GridWorldBase):
    """Environment for training agents to reach bottom-right (alternative team goal)."""

    def _set_goal_position(self):
        """Set goal to bottom-right corner."""
        self.goal_pos = np.array([self.size - 1, self.size - 1], dtype=np.int32)

    def _calculate_reward(self) -> float:
        """Reward for reaching bottom-right corner."""
        if self._is_goal_reached():
            return 10.0
        distance = np.linalg.norm(self.agent_pos.astype(float) - self.goal_pos.astype(float))
        return -(STEP_PENALTY + DISTANCE_WEIGHT * distance)


class TeamGoalBottomLeft(GridWorldBase):
    """Environment for training agents to reach bottom-left (alternative team goal)."""

    def _set_goal_position(self):
        """Set goal to bottom-left corner."""
        self.goal_pos = np.array([0, self.size - 1], dtype=np.int32)

    def _calculate_reward(self) -> float:
        """Reward for reaching bottom-left corner."""
        if self._is_goal_reached():
            return 10.0
        distance = np.linalg.norm(self.agent_pos.astype(float) - self.goal_pos.astype(float))
        return -(STEP_PENALTY + DISTANCE_WEIGHT * distance)
