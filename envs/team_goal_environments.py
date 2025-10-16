"""Single-agent environments for training team-specific goal policies.

These environments are used to train individual agents for each team's goal.
During training, we train policies for:
- Team goals (e.g., reach top-right for Team A, reach top-left for Team B)

In the multi-agent recognition phase, we'll match observed behaviors
against these trained policies to determine team membership and goals.
"""

from .grid_world import GridWorldBase
import numpy as np
from typing import Optional, Tuple, Dict


class TeamGoalTopRight(GridWorldBase):
    """Environment for training agents to reach top-right (Team A's goal)."""

    def _set_goal_position(self):
        """Set goal to top-right corner."""
        self.goal_pos = np.array([self.size - 1, 0], dtype=np.int32)

    def _calculate_reward(self) -> float:
        """Reward for reaching top-right corner."""
        if self._is_goal_reached():
            return 10.0  # Big reward for reaching goal
        else:
            # Small negative reward proportional to distance
            distance = np.linalg.norm(self.agent_pos.astype(float) - self.goal_pos.astype(float))
            return -0.01 * distance


class TeamGoalTopLeft(GridWorldBase):
    """Environment for training agents to reach top-left (Team B's goal)."""

    def _set_goal_position(self):
        """Set goal to top-left corner."""
        self.goal_pos = np.array([0, 0], dtype=np.int32)

    def _calculate_reward(self) -> float:
        """Reward for reaching top-left corner."""
        if self._is_goal_reached():
            return 10.0
        else:
            distance = np.linalg.norm(self.agent_pos.astype(float) - self.goal_pos.astype(float))
            return -0.01 * distance


class TeamGoalBottomRight(GridWorldBase):
    """Environment for training agents to reach bottom-right (alternative team goal)."""

    def _set_goal_position(self):
        """Set goal to bottom-right corner."""
        self.goal_pos = np.array([self.size - 1, self.size - 1], dtype=np.int32)

    def _calculate_reward(self) -> float:
        """Reward for reaching bottom-right corner."""
        if self._is_goal_reached():
            return 10.0
        else:
            distance = np.linalg.norm(self.agent_pos.astype(float) - self.goal_pos.astype(float))
            return -0.01 * distance


class TeamGoalBottomLeft(GridWorldBase):
    """Environment for training agents to reach bottom-left (alternative team goal)."""

    def _set_goal_position(self):
        """Set goal to bottom-left corner."""
        self.goal_pos = np.array([0, self.size - 1], dtype=np.int32)

    def _calculate_reward(self) -> float:
        """Reward for reaching bottom-left corner."""
        if self._is_goal_reached():
            return 10.0
        else:
            distance = np.linalg.norm(self.agent_pos.astype(float) - self.goal_pos.astype(float))
            return -0.01 * distance


class TeamGoalCenter(GridWorldBase):
    """Environment for training agents to reach/stay at center (alternative team goal)."""

    def _set_goal_position(self):
        """Set goal to center."""
        self.goal_pos = np.array([self.size // 2, self.size // 2], dtype=np.int32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset - for center goal, we allow starting at the goal."""
        super(GridWorldBase, self).reset(seed=seed)

        # Set goal position
        self._set_goal_position()

        # For stay-at-center goal, start at random position (can be center)
        random_x = self.np_random.integers(0, self.size)
        random_y = self.np_random.integers(0, self.size)
        self.agent_pos = np.array([random_x, random_y], dtype=np.int32)

        self.steps_taken = 0

        if self.render_mode == 'console':
            self.render()

        return self.agent_pos.copy(), {}

    def _calculate_reward(self) -> float:
        """Reward for staying at center."""
        if self._is_goal_reached():
            return 1.0  # Constant reward for being at center
        else:
            # Negative reward for being away from center
            distance = np.linalg.norm(self.agent_pos.astype(float) - self.goal_pos.astype(float))
            return -0.1 * distance