from __future__ import annotations

import math
import heapq
from typing import Dict, Tuple, Optional, Sequence, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .scenario import Scenario
from .runtime import RuntimeParameters


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


class StarCraftTeamEnv(gym.Env):
    """
    Multi-agent wrapper that jointly trains a policy for multiple StarCraft scenarios.

    The team environment concatenates per-agent observations from StarCraftScenarioEnv
    instances and exposes a MultiDiscrete action space where each entry controls one
    agent directly. The episode terminates successfully once all agents have reached
    their respective goals. Episodes truncate when the shared step budget is
    exhausted or when all agents finish without collectively succeeding.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        grid: np.ndarray,
        scenarios: Sequence[Scenario],
        runtimes: Sequence[RuntimeParameters],
        team_name: str = "team",
        reward_mode: str = "sum",
    ) -> None:
        if not scenarios:
            raise ValueError("StarCraftTeamEnv requires at least one agent scenario")
        if len(scenarios) != len(runtimes):
            raise ValueError("Number of scenarios must match number of runtime configs")
        if grid.dtype != bool:
            raise ValueError("grid must be a boolean numpy array indicating walkable tiles")

        self.team_name = team_name
        self.reward_mode = reward_mode.strip().lower()
        if self.reward_mode not in {"sum", "mean"}:
            raise ValueError("reward_mode must be either 'sum' or 'mean'")

        self.num_agents = len(scenarios)
        self._agent_envs: List[StarCraftScenarioEnv] = []
        for scenario, runtime in zip(scenarios, runtimes):
            env = StarCraftScenarioEnv(
                grid=grid,
                scenario=scenario,
                max_steps=runtime.max_steps,
                step_penalty=runtime.step_penalty,
                invalid_penalty=runtime.invalid_penalty,
                goal_reward=runtime.goal_reward,
                progress_scale=runtime.progress_scale,
                view_radius=runtime.view_radius,
                allow_diagonal_actions=runtime.allow_diagonal_actions,
                view_mode=runtime.view_mode,
            )
            self._agent_envs.append(env)

        self._agent_obs: List[np.ndarray] = []
        self._agent_done: List[bool] = []
        self._agent_success: List[bool] = []
        self._agent_last_info: List[Dict[str, object]] = []
        self._agent_positions: List[Coordinate] = []

        # Observation space concatenates each per-agent observation vector.
        obs_dim = sum(env.observation_space.shape[0] for env in self._agent_envs)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        base_actions = self._agent_envs[0].action_space
        if not isinstance(base_actions, spaces.Discrete):
            raise ValueError("Team env expects underlying scenarios to use Discrete actions")
        self._joint_action_base = base_actions.n
        self.action_space = spaces.MultiDiscrete(
            np.full(self.num_agents, self._joint_action_base, dtype=np.int64)
        )

        self.max_steps = max(runtime.max_steps for runtime in runtimes)
        self._steps = 0

    # Gym API -------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._steps = 0
        self._agent_done = [False] * self.num_agents
        self._agent_success = [False] * self.num_agents
        self._agent_obs = []
        self._agent_last_info = []
        self._agent_positions = []

        for idx, env in enumerate(self._agent_envs):
            env_seed = seed if idx == 0 else None
            obs, info = env.reset(seed=env_seed)
            self._agent_obs.append(np.array(obs, dtype=np.float32))
            start_pos = tuple(info.get("position", env.scenario.start))
            self._agent_positions.append(start_pos)
            payload = dict(info or {})
            payload.setdefault("position", start_pos)
            self._agent_last_info.append(payload)

        return self._concat_obs(), {"team_name": self.team_name}

    def step(self, action):
        self._steps += 1
        decoded_actions = self._prepare_actions(action)
        total_reward = 0.0
        agent_infos: List[Dict[str, object]] = []

        for idx, env in enumerate(self._agent_envs):
            if self._agent_done[idx]:
                cached = dict(self._agent_last_info[idx])
                cached.setdefault("position", self._agent_positions[idx])
                cached.setdefault("terminated", self._agent_success[idx])
                cached.setdefault("truncated", not self._agent_success[idx])
                cached["skipped"] = True
                agent_infos.append(cached)
                continue

            obs, reward, terminated, truncated, info = env.step(decoded_actions[idx])
            self._agent_obs[idx] = np.array(obs, copy=True)
            total_reward += float(reward)
            current_position = tuple(info.get("position", self._agent_positions[idx]))
            self._agent_positions[idx] = current_position

            if terminated:
                self._agent_done[idx] = True
                self._agent_success[idx] = True
            elif truncated:
                self._agent_done[idx] = True

            payload = dict(info or {})
            payload["terminated"] = terminated
            payload["truncated"] = truncated
            payload.setdefault("position", current_position)
            self._agent_last_info[idx] = payload
            agent_infos.append(payload)

        if self.reward_mode == "mean":
            total_reward /= float(self.num_agents)

        all_finished = all(self._agent_done)
        team_success = all_finished and all(self._agent_success)
        terminated = team_success
        truncated = False

        if self._steps >= self.max_steps:
            truncated = True
        elif all_finished and not team_success:
            truncated = True

        info = {
            "team_name": self.team_name,
            "agent_info": agent_infos,
            "team_success": team_success,
            "steps": self._steps,
        }

        return self._concat_obs(), float(total_reward), terminated, truncated, info

    def close(self):
        for env in self._agent_envs:
            env.close()
        super().close()

    # Internals -----------------------------------------------------------
    def _concat_obs(self) -> np.ndarray:
        if not self._agent_obs:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.concatenate(self._agent_obs).astype(np.float32, copy=False)

    def _prepare_actions(self, action) -> List[int]:
        if isinstance(action, (list, tuple, np.ndarray)):
            arr = np.asarray(action, dtype=np.int64)
            if arr.shape != (self.num_agents,):
                raise ValueError(
                    f"Expected action shape ({self.num_agents},), received {arr.shape}"
                )
            return arr.tolist()
        return self._decode_joint_action(int(action))

    def _decode_joint_action(self, action: int) -> List[int]:
        action = int(action)
        max_joint = int(self._joint_action_base ** self.num_agents) - 1
        action = max(0, min(action, max_joint))
        decoded: List[int] = []
        base = self._joint_action_base
        remainder = action
        for _ in range(self.num_agents):
            decoded.append(remainder % base)
            remainder //= base
        return decoded
