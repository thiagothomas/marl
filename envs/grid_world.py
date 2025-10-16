"""Custom grid world environment for goal recognition experiments."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class GridWorldBase(gym.Env):
    """Base grid world environment for goal recognition.

    This is a simple grid world where an agent navigates to different goal positions.
    Each subclass defines a different goal position, encoding different objectives.
    """

    metadata = {'render_modes': ['console']}

    def __init__(self, size: int = 7, max_steps: int = 100, render_mode: Optional[str] = None):
        """Initialize the grid world.

        Args:
            size: Size of the square grid
            max_steps: Maximum number of steps per episode
            render_mode: How to render the environment
        """
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 4 discrete actions: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Observation is the (x, y) position of the agent
        self.observation_space = spaces.Box(
            low=0,
            high=size-1,
            shape=(2,),
            dtype=np.int32
        )

        # Actions mapping
        self.actions = {
            0: np.array([0, -1]),  # Up
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([1, 0])    # Right
        }

        # Initialize state variables
        self.agent_pos = None
        self.goal_pos = None
        self.steps_taken = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state.

        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)

        # Set goal position first (defined in subclasses)
        self._set_goal_position()

        # Start agent at random position (but not on the goal)
        while True:
            random_x = self.np_random.integers(0, self.size)
            random_y = self.np_random.integers(0, self.size)
            self.agent_pos = np.array([random_x, random_y], dtype=np.int32)

            # Make sure we don't start on the goal
            if not np.array_equal(self.agent_pos, self.goal_pos):
                break

        self.steps_taken = 0

        if self.render_mode == 'console':
            self.render()

        return self.agent_pos.copy(), {}

    def _set_goal_position(self):
        """Set the goal position. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _set_goal_position")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.

        Args:
            action: Action to take (0=up, 1=down, 2=left, 3=right)

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode is done (reached goal)
            truncated: Whether episode is truncated (max steps)
            info: Additional information
        """
        # Move agent
        new_pos = self.agent_pos + self.actions[action]

        # Clip to grid boundaries
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.agent_pos = new_pos

        self.steps_taken += 1

        # Calculate reward (defined in subclasses)
        reward = self._calculate_reward()

        # Check if goal is reached
        terminated = self._is_goal_reached()

        # Check if max steps exceeded
        truncated = self.steps_taken >= self.max_steps

        if self.render_mode == 'console':
            self.render()

        return self.agent_pos.copy(), reward, terminated, truncated, {}

    def _calculate_reward(self) -> float:
        """Calculate reward for current state. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _calculate_reward")

    def _is_goal_reached(self) -> bool:
        """Check if the goal has been reached."""
        return np.array_equal(self.agent_pos, self.goal_pos)

    def render(self):
        """Render the environment to console."""
        if self.render_mode == 'console':
            grid = np.full((self.size, self.size), '.')
            grid[self.goal_pos[1], self.goal_pos[0]] = 'G'
            grid[self.agent_pos[1], self.agent_pos[0]] = 'A'

            print("\n" + "=" * (self.size * 2 + 1))
            for row in grid:
                print(' '.join(row))
            print("=" * (self.size * 2 + 1))
            print(f"Steps: {self.steps_taken}/{self.max_steps}")
