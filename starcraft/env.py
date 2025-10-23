from __future__ import annotations

import math
import heapq
from typing import Dict, Tuple, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .scenario import Scenario


Action = int
Coordinate = Tuple[int, int]


CARDINAL_DELTAS: Tuple[Coordinate, ...] = (
    (0, -1),   # N
    (1, 0),    # E
    (0, 1),    # S
    (-1, 0),   # W
)

DIAGONAL_DELTAS: Tuple[Coordinate, ...] = (
    (1, -1),   # NE
    (1, 1),    # SE
    (-1, 1),   # SW
    (-1, -1),  # NW
)


class StarCraftScenarioEnv(gym.Env):
    """
    Lightweight single-agent environment built from a Moving AI StarCraft map.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid: np.ndarray,
        scenario: Scenario,
        *,
        max_steps: Optional[int] = None,
        max_steps_scale: float = 4.0,
        step_penalty: Optional[float] = None,
        invalid_penalty: Optional[float] = None,
        goal_reward: float = 1.0,
        progress_scale: float = 0.1,
        view_radius: int = 1,
        allow_diagonal_actions: bool = False,
        view_mode: str = "cardinal",
    ) -> None:
        if grid.dtype != bool:
            raise ValueError("grid must be a boolean numpy array indicating walkable tiles")

        self.grid = grid
        self.scenario = scenario
        self.height, self.width = grid.shape
        self.goal_reward = goal_reward
        self.progress_scale = progress_scale
        self.view_mode = view_mode.strip().lower()

        if allow_diagonal_actions:
            self._action_deltas: Sequence[Coordinate] = CARDINAL_DELTAS + DIAGONAL_DELTAS
        else:
            self._action_deltas = CARDINAL_DELTAS
        self._action_map: Dict[Action, Coordinate] = {
            idx: delta for idx, delta in enumerate(self._action_deltas)
        }

        if self.view_mode == "cardinal":
            self.view_radius = 1
            self.view_diameter = 1
            self.view_area = 4
        elif self.view_mode in {"moore", "3x3"}:
            self.view_radius = 1
            self.view_diameter = 3
            self.view_area = 9
        elif self.view_mode in {"window", "square", "full"}:
            self.view_radius = max(int(view_radius), 0)
            self.view_diameter = self.view_radius * 2 + 1
            self.view_area = self.view_diameter * self.view_diameter
        else:
            raise ValueError(
                "view_mode must be one of {'cardinal', 'moore', '3x3', 'window', 'square', 'full'}"
            )

        if max_steps is None:
            computed_steps = max_steps_scale * max(1.0, scenario.optimal_length)
            self.max_steps = max(int(round(computed_steps)), 1)
        else:
            self.max_steps = max(int(max_steps), 1)

        if step_penalty is None:
            self.step_penalty = 0.01
        else:
            self.step_penalty = float(step_penalty)
        if invalid_penalty is None:
            self.invalid_penalty = 4.0 * self.step_penalty
        else:
            self.invalid_penalty = float(invalid_penalty)

        self.action_space = spaces.Discrete(len(self._action_deltas))
        obs_size = 4 + self.view_area
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self._position = np.array([scenario.start[0], scenario.start[1]], dtype=np.int32)
        self._steps = 0
        self._distance_map = self._compute_distance_map()
        self._current_distance = self._distance_at(self._position[0], self._position[1])

    # Gym API -------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._position[:] = (self.scenario.start[0], self.scenario.start[1])
        self._steps = 0
        self._current_distance = self._distance_at(self._position[0], self._position[1])
        return self._get_obs(), {"position": tuple(self._position)}

    def step(self, action: Action):
        self._steps += 1
        dx, dy = self._action_map.get(int(action), (0, 0))
        new_x = int(self._position[0] + dx)
        new_y = int(self._position[1] + dy)

        reward = 0.0
        terminated = False
        truncated = False

        valid_move = self._valid_position(new_x, new_y)

        if not valid_move:
            reward -= self.invalid_penalty
        else:
            self._position[:] = (new_x, new_y)
            reward -= self.step_penalty

            new_distance = self._distance_at(new_x, new_y)
            if np.isfinite(new_distance) and np.isfinite(self._current_distance):
                progress = self._current_distance - new_distance
                reward += self.progress_scale * progress
                self._current_distance = new_distance

            if (new_x, new_y) == self.scenario.goal:
                reward += self.goal_reward
                terminated = True

        if self._steps >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info = {"position": tuple(self._position), "valid_move": valid_move}
        return obs, reward, terminated, truncated, info

    # Internals -----------------------------------------------------------
    def _get_obs(self):
        px = self._position[0] / (self.width - 1)
        py = self._position[1] / (self.height - 1)
        gx = self.scenario.goal[0] / (self.width - 1)
        gy = self.scenario.goal[1] / (self.height - 1)
        local_patch = self._get_local_patch(self._position[0], self._position[1])
        return np.concatenate((np.array([px, py, gx, gy], dtype=np.float32), local_patch))

    def _valid_position(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return bool(self.grid[y, x])

    # Distance map -------------------------------------------------------
    def _compute_distance_map(self) -> np.ndarray:
        goal_x, goal_y = self.scenario.goal
        distance = np.full((self.height, self.width), np.inf, dtype=np.float32)
        if not self._valid_position(goal_x, goal_y):
            return distance

        distance[goal_y, goal_x] = 0.0
        heap: list[Tuple[float, int, int]] = [(0.0, goal_x, goal_y)]

        while heap:
            dist, x, y = heapq.heappop(heap)
            if dist > distance[y, x]:
                continue

            for dx, dy in self._action_deltas:
                nx, ny = x + dx, y + dy
                if not self._valid_position(nx, ny):
                    continue

                step_cost = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
                new_dist = dist + step_cost
                if new_dist < distance[ny, nx]:
                    distance[ny, nx] = new_dist
                    heapq.heappush(heap, (new_dist, nx, ny))

        return distance

    def _distance_at(self, x: int, y: int) -> float:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return float("inf")
        return float(self._distance_map[y, x])

    def _get_local_patch(self, center_x: int, center_y: int) -> np.ndarray:
        """
        Flattened occupancy window around the agent. Walkable tiles are 1.0, blocked or out-of-bounds are 0.0.
        """
        if self.view_mode == "cardinal":
            patch = np.zeros(4, dtype=np.float32)
            for idx, (dx, dy) in enumerate(CARDINAL_DELTAS):
                x = center_x + dx
                y = center_y + dy
                if 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x]:
                    patch[idx] = 1.0
            return patch

        if self.view_mode in {"moore", "3x3"}:
            patch = np.zeros((3, 3), dtype=np.float32)
            for local_y, dy in enumerate(range(-1, 2)):
                for local_x, dx in enumerate(range(-1, 2)):
                    x = center_x + dx
                    y = center_y + dy
                    if 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x]:
                        patch[local_y, local_x] = 1.0
            return patch.reshape(-1)

        if self.view_radius == 0:
            if 0 <= center_x < self.width and 0 <= center_y < self.height and self.grid[center_y, center_x]:
                return np.array([1.0], dtype=np.float32)
            return np.array([0.0], dtype=np.float32)

        patch = np.zeros((self.view_diameter, self.view_diameter), dtype=np.float32)
        for dy in range(-self.view_radius, self.view_radius + 1):
            for dx in range(-self.view_radius, self.view_radius + 1):
                x = center_x + dx
                y = center_y + dy
                if 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x]:
                    patch[dy + self.view_radius, dx + self.view_radius] = 1.0
        return patch.reshape(-1)
