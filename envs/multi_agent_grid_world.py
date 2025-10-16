"""Multi-agent grid world environment for team-based goal recognition."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from .grid_world import GridWorldBase


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
        render_mode: Optional[str] = None
    ):
        """Initialize the multi-agent grid world.

        Args:
            size: Size of the square grid
            max_steps: Maximum number of steps per episode
            team_sizes: List of number of agents per team [team1_size, team2_size, ...]
            team_goals: List of goal types for each team ['top_right', 'top_left', ...]
            render_mode: How to render the environment
        """
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode

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

        # Define goal positions for each team
        self.goal_positions = self._define_goal_positions()

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
            elif goal_type == 'center':
                goal_positions[team_id] = np.array([self.size // 2, self.size // 2], dtype=np.int32)
            else:
                raise ValueError(f"Unknown goal type: {goal_type}")

        return goal_positions

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state.

        Returns:
            observations: Initial observations for all agents
            info: Additional information
        """
        super().reset(seed=seed)

        # Initialize agent positions randomly (avoiding goal positions)
        self.agent_positions = {}
        occupied_positions = set()

        for agent_id in range(self.num_agents):
            while True:
                x = self.np_random.integers(0, self.size)
                y = self.np_random.integers(0, self.size)
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
            'steps_taken': self.steps_taken
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

            # Check for collisions with other agents (optional: can disable)
            collision = False
            for other_id, other_pos in self.agent_positions.items():
                if other_id != agent_id and np.array_equal(new_pos, other_pos):
                    collision = True
                    break

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
                reward = -0.01 * distance

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


# Convenience classes for specific multi-agent scenarios
class TwoTeamsSingleAgent(MultiAgentGridWorld):
    """Two teams with one agent each - Team 0 goes top-right, Team 1 goes top-left."""

    def __init__(self, size: int = 7, max_steps: int = 100, render_mode: Optional[str] = None):
        super().__init__(
            size=size,
            max_steps=max_steps,
            team_sizes=[1, 1],
            team_goals=['top_right', 'top_left'],
            render_mode=render_mode
        )


class TwoTeamsDoubleAgents(MultiAgentGridWorld):
    """Two teams with two agents each - Team 0 goes top-right, Team 1 goes top-left."""

    def __init__(self, size: int = 7, max_steps: int = 100, render_mode: Optional[str] = None):
        super().__init__(
            size=size,
            max_steps=max_steps,
            team_sizes=[2, 2],
            team_goals=['top_right', 'top_left'],
            render_mode=render_mode
        )