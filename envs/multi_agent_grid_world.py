"""Multi-agent grid world environment for team-based goal recognition."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from .grid_world import GridWorldBase

DEFAULT_INITIAL_POSITION_PRESETS = {
    (1, 1): [
        [(1, 6), (5, 6)],
        [(1, 5), (5, 5)],
    ],
    (2, 2): [
        [(1, 6), (1, 5), (5, 6), (5, 5)],
    ],
}


def _select_default_initial_positions(
    team_sizes: Tuple[int, ...],
    preset_index: int = 0
) -> Optional[List[Tuple[int, int]]]:
    """Return a default initial position preset for the given team sizes."""
    presets = DEFAULT_INITIAL_POSITION_PRESETS.get(team_sizes)
    if not presets:
        return None

    preset_index = max(0, min(preset_index, len(presets) - 1))
    return presets[preset_index]


class MultiAgentGridWorld(gym.Env):
    """Multi-agent grid world environment where teams of agents pursue different goals.

    This environment supports multiple teams, where each team has:
    - One or more agents
    - A shared team goal (e.g., reach top-right, reach top-left)
    - Independent movement but shared objective

    The environment maintains separate positions for each agent but evaluates
    success based on team goals.
    """

    metadata = {'render_modes': ['console', 'rgb_array']}

    def __init__(
        self,
        size: int = 7,
        max_steps: int = 100,
        team_sizes: List[int] = None,
        team_goals: List[str] = None,
        render_mode: Optional[str] = None,
        initial_agent_positions: Optional[List[Tuple[int, int]]] = None,
        use_default_start_positions: bool = True,
        start_preset: int = 0,
        step_penalty: float = 0.01
    ):
        """Initialize the multi-agent grid world.

        Args:
            size: Size of the square grid
            max_steps: Maximum number of steps per episode
            team_sizes: List of number of agents per team [team1_size, team2_size, ...]
            team_goals: List of goal types for each team ['top_right', 'top_left', ...]
            render_mode: How to render the environment
            step_penalty: Constant per-step penalty applied until success
        """
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.use_default_start_positions = use_default_start_positions
        self.start_preset = start_preset
        self.step_penalty = float(step_penalty)

        # Default: 2 teams with 1 agent each
        self.team_sizes = team_sizes if team_sizes is not None else [1, 1]
        self.team_goals = team_goals if team_goals is not None else ['top_right', 'top_left']

        # Validate inputs
        assert len(self.team_sizes) == len(self.team_goals), "Number of teams must match number of goals"

        # Calculate total number of agents
        self.num_teams = len(self.team_sizes)
        self.num_agents = sum(self.team_sizes)

        # Create agent-to-team mapping
        self.agent_teams = []  # Maps agent_id to team_id
        for team_id, team_size in enumerate(self.team_sizes):
            self.agent_teams.extend([team_id] * team_size)

        self.initial_agent_positions = None
        if initial_agent_positions is not None:
            self.initial_agent_positions = self._normalize_initial_positions(initial_agent_positions)
        elif self.use_default_start_positions:
            default_positions = _select_default_initial_positions(tuple(self.team_sizes), self.start_preset)
            if default_positions is not None:
                self.initial_agent_positions = self._normalize_initial_positions(default_positions)

        # Define goal positions for each team
        self.goal_positions = self._define_goal_positions()
        self.obstacles = self._define_obstacles()
        self._obstacle_set = {tuple(pos) for pos in self.obstacles}

        # Action space: each agent can move independently (4 actions per agent)
        # We'll use a dict action space for clarity
        self.action_space = spaces.Dict({
            f'agent_{i}': spaces.Discrete(4) for i in range(self.num_agents)
        })

        # Observation space: position of all agents (for full observability)
        # Each agent observes all agent positions
        self.observation_space = spaces.Dict({
            f'agent_{i}': spaces.Box(
                low=0, high=size-1, shape=(self.num_agents * 2,), dtype=np.int32
            ) for i in range(self.num_agents)
        })

        # Actions mapping
        self.actions_map = {
            0: np.array([0, -1]),  # Up
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([1, 0])    # Right
        }

        # State variables
        self.agent_positions = None
        self.steps_taken = 0
        self.team_success = None  # Track which teams have reached their goals

    def _define_goal_positions(self) -> Dict[int, np.ndarray]:
        """Define goal positions for each team based on goal type."""
        goal_positions = {}

        for team_id, goal_type in enumerate(self.team_goals):
            if goal_type == 'top_right':
                goal_positions[team_id] = np.array([self.size - 1, 0], dtype=np.int32)
            elif goal_type == 'top_left':
                goal_positions[team_id] = np.array([0, 0], dtype=np.int32)
            elif goal_type == 'bottom_left':
                goal_positions[team_id] = np.array([0, self.size - 1], dtype=np.int32)
            elif goal_type == 'bottom_right':
                goal_positions[team_id] = np.array([self.size - 1, self.size - 1], dtype=np.int32)
            else:
                raise ValueError(f"Unknown goal type: {goal_type}")

        return goal_positions

    def _normalize_initial_positions(self, positions: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Validate and convert initial positions into numpy arrays."""
        if len(positions) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} initial positions, got {len(positions)}"
            )

        normalized: List[np.ndarray] = []
        for idx, pos in enumerate(positions):
            arr = np.asarray(pos, dtype=np.int32)
            if arr.shape != (2,):
                raise ValueError(
                    f"Initial position for agent {idx} must have 2 coordinates, got {arr}"
                )
            if np.any(arr < 0) or np.any(arr >= self.size):
                raise ValueError(
                    f"Initial position {tuple(arr.tolist())} for agent {idx} is out of bounds"
                )
            normalized.append(arr)

        return normalized

    def _define_obstacles(self) -> List[Tuple[int, int]]:
        """Static obstacle layout shared across multi-agent scenarios."""
        mid = self.size // 2
        obstacles: List[Tuple[int, int]] = []

        for y in range(1, self.size - 1):
            if y == mid:
                continue
            obstacles.append((mid, y))

        for x in range(1, self.size - 1):
            if x == mid:
                continue
            obstacles.append((x, mid))

        extras = [
            (mid - 1, 1),
            (mid + 1, self.size - 2)
        ]

        for pos in extras:
            if 0 <= pos[0] < self.size and 0 <= pos[1] < self.size:
                obstacles.append(pos)

        return obstacles

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state.

        Returns:
            observations: Initial observations for all agents
            info: Additional information
        """
        super().reset(seed=seed)

        # Initialize agent positions (deterministic presets or random fallback)
        self.agent_positions = {}
        obstacle_positions = self._obstacle_set.copy()
        occupied_positions = set(obstacle_positions)

        if self.initial_agent_positions is not None:
            for agent_id in range(self.num_agents):
                start_pos = self.initial_agent_positions[agent_id]
                pos_tuple = (int(start_pos[0]), int(start_pos[1]))

                if pos_tuple in obstacle_positions:
                    raise ValueError(
                        f"Initial position {pos_tuple} overlaps an obstacle"
                    )
                if pos_tuple in occupied_positions:
                    raise ValueError(
                        f"Duplicate initial position detected for agent {agent_id}: {pos_tuple}"
                    )

                self.agent_positions[agent_id] = start_pos.copy()
                occupied_positions.add(pos_tuple)
        else:
            for agent_id in range(self.num_agents):
                while True:
                    x = int(self.np_random.integers(0, self.size))
                    y = int(self.np_random.integers(0, self.size))
                    pos = (x, y)

                    # Check if position is free and not a goal
                    is_goal = any(
                        np.array_equal([x, y], goal_pos)
                        for goal_pos in self.goal_positions.values()
                    )

                    if pos not in occupied_positions and not is_goal:
                        self.agent_positions[agent_id] = np.array([x, y], dtype=np.int32)
                        occupied_positions.add(pos)
                        break

        self.steps_taken = 0
        self.team_success = {team_id: False for team_id in range(self.num_teams)}

        if self.render_mode == 'console':
            self.render()

        return self._get_observations(), self._get_info()

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents.

        Each agent observes all agent positions (full observability).
        """
        # Flatten all agent positions into a single array
        all_positions = []
        for agent_id in range(self.num_agents):
            all_positions.extend(self.agent_positions[agent_id].tolist())
        all_positions = np.array(all_positions, dtype=np.int32)

        # Each agent gets the same observation (full observability)
        observations = {}
        for agent_id in range(self.num_agents):
            observations[f'agent_{agent_id}'] = all_positions.copy()

        return observations

    def _get_info(self) -> Dict:
        """Get additional environment information."""
        info = {
            'team_success': self.team_success.copy(),
            'agent_teams': self.agent_teams.copy(),
            'goal_positions': {k: v.copy() for k, v in self.goal_positions.items()},
            'steps_taken': self.steps_taken,
            'obstacles': list(self.obstacles),
            'initial_positions': [
                pos.tolist() for pos in self.initial_agent_positions
            ] if self.initial_agent_positions is not None else None
        }
        return info

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute actions for all agents.

        Args:
            actions: Dict mapping agent_id to action

        Returns:
            observations: New observations for all agents
            rewards: Rewards for all agents
            terminateds: Whether each agent's episode is done
            truncateds: Whether each agent's episode is truncated
            info: Additional information
        """
        # Move each agent
        for agent_id in range(self.num_agents):
            action = actions.get(f'agent_{agent_id}', 0)  # Default to no-op if missing

            # Calculate new position
            new_pos = self.agent_positions[agent_id] + self.actions_map[action]

            # Clip to grid boundaries
            new_pos = np.clip(new_pos, 0, self.size - 1)

            # Only check for collisions with obstacles (agents can occupy same square)
            collision = (new_pos[0], new_pos[1]) in self._obstacle_set

            if not collision:
                self.agent_positions[agent_id] = new_pos

        self.steps_taken += 1

        # Calculate rewards and check termination for each team
        rewards = {}
        terminateds = {}

        for agent_id in range(self.num_agents):
            team_id = self.agent_teams[agent_id]
            goal_pos = self.goal_positions[team_id]

            # Check if this agent has reached the team's goal
            at_goal = np.array_equal(self.agent_positions[agent_id], goal_pos)

            if at_goal and not self.team_success[team_id]:
                # First agent from team to reach goal
                self.team_success[team_id] = True
                reward = 10.0  # Big reward for reaching goal
            elif self.team_success[team_id]:
                # Team already succeeded
                reward = 0.0
            else:
                # Negative reward based on distance to goal
                distance = np.linalg.norm(
                    self.agent_positions[agent_id].astype(float) - goal_pos.astype(float)
                )
                reward = -(self.step_penalty + 0.01 * distance)

            rewards[f'agent_{agent_id}'] = reward
            terminateds[f'agent_{agent_id}'] = self.team_success[team_id]

        # Check truncation (max steps)
        truncated = self.steps_taken >= self.max_steps
        truncateds = {f'agent_{agent_id}': truncated for agent_id in range(self.num_agents)}

        if self.render_mode == 'console':
            self.render()

        return self._get_observations(), rewards, terminateds, truncateds, self._get_info()

    def render(self):
        """Render the environment to console."""
        if self.render_mode == 'console':
            # Create grid
            grid = np.full((self.size, self.size), '.', dtype=str)

            # Mark goals with team IDs
            for team_id, goal_pos in self.goal_positions.items():
                grid[goal_pos[1], goal_pos[0]] = f'G{team_id}'

            # Mark obstacles
            for ox, oy in self.obstacles:
                grid[oy, ox] = '##'

            # Mark agents with their IDs and team colors
            team_symbols = ['A', 'B', 'C', 'D', 'E']  # Different symbols for teams
            for agent_id, pos in self.agent_positions.items():
                team_id = self.agent_teams[agent_id]
                symbol = team_symbols[team_id % len(team_symbols)]
                # Show agent number within team
                agent_num_in_team = sum(1 for i in range(agent_id) if self.agent_teams[i] == team_id)
                grid[pos[1], pos[0]] = f'{symbol}{agent_num_in_team}'

            # Print grid
            print("\n" + "=" * (self.size * 3 + 1))
            for row in grid:
                print(' '.join(f'{cell:2}' for cell in row))
            print("=" * (self.size * 3 + 1))
            print(f"Steps: {self.steps_taken}/{self.max_steps}")
            print(f"Team Success: {self.team_success}")

            # Print legend
            print("\nLegend:")
            for team_id in range(self.num_teams):
                symbol = team_symbols[team_id % len(team_symbols)]
                goal = self.team_goals[team_id]
                print(f"  Team {team_id} ({symbol}): Goal = {goal} at {self.goal_positions[team_id]}")
            print(f"  Obstacles (#): {sorted(self.obstacles)}")


# Convenience classes for specific multi-agent scenarios
class TwoTeamsSingleAgent(MultiAgentGridWorld):
    """Two teams with one agent each - Team 0 goes top-right, Team 1 goes top-left."""

    def __init__(
        self,
        size: int = 7,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        initial_agent_positions: Optional[List[Tuple[int, int]]] = None,
        use_default_start_positions: bool = True,
        start_preset: int = 0
    ):
        super().__init__(
            size=size,
            max_steps=max_steps,
            team_sizes=[1, 1],
            team_goals=['top_right', 'top_left'],
            render_mode=render_mode,
            initial_agent_positions=initial_agent_positions,
            use_default_start_positions=use_default_start_positions,
            start_preset=start_preset
        )


class TwoTeamsDoubleAgents(MultiAgentGridWorld):
    """Two teams with two agents each - Team 0 goes top-right, Team 1 goes top-left."""

    def __init__(
        self,
        size: int = 7,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        initial_agent_positions: Optional[List[Tuple[int, int]]] = None,
        use_default_start_positions: bool = True,
        start_preset: int = 0
    ):
        super().__init__(
            size=size,
            max_steps=max_steps,
            team_sizes=[2, 2],
            team_goals=['top_right', 'top_left'],
            render_mode=render_mode,
            initial_agent_positions=initial_agent_positions,
            use_default_start_positions=use_default_start_positions,
            start_preset=start_preset
        )
