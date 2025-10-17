#!/usr/bin/env python3
"""
Multi-Agent Incremental Goal Recognition Visualizer.

Shows step-by-step how team assignments and goal beliefs update as multiple
agents take actions simultaneously.
"""

import os
import sys
import time
import numpy as np
from itertools import combinations, product
from typing import Any, Dict, List, Tuple

# Add parent directory to path

from envs.team_goal_environments import (
    TeamGoalTopRight,
    TeamGoalTopLeft,
    TeamGoalBottomLeft,
    TeamGoalBottomRight
)
from envs.multi_agent_grid_world import MultiAgentGridWorld, DEFAULT_INITIAL_POSITION_PRESETS
from ml.ppo import PPOAgent
from metrics.metrics import kl_divergence, softmin


class MultiAgentIncrementalVisualizer:
    """Visualize incremental multi-agent goal and team recognition."""

    def __init__(self, agents, goal_names, evaluation_function, num_agents, team_sizes):
        self.agents = agents
        self.goal_names = goal_names
        self.evaluation_function = evaluation_function
        self.num_agents = num_agents
        self.team_sizes = team_sizes
        self.num_teams = len(team_sizes)

    def generate_goal_combinations(self) -> List[Tuple[str, ...]]:
        """Generate all possible combinations of team goals."""
        return list(product(self.goal_names, repeat=self.num_teams))

    def _score_team_assignments(
        self,
        observations_per_agent,
        possible_team_goals,
        team_partitions
    ):
        """Internal helper mirroring the recognizer scoring logic."""
        ranked = []

        for partition in team_partitions:
            for goal_combination in possible_team_goals:
                if len(goal_combination) != len(partition):
                    continue

                total_score = 0.0
                assignment = {}
                score_details = []
                valid_combo = True

                for team_id, team_agents in enumerate(partition):
                    team_goal = goal_combination[team_id]
                    try:
                        goal_idx = self.goal_names.index(team_goal)
                    except ValueError:
                        valid_combo = False
                        break

                    policy_agent = self.agents[goal_idx]

                    for agent_idx in team_agents:
                        agent_id = f'agent_{agent_idx}'
                        agent_obs = observations_per_agent.get(agent_id, [])

                        if agent_obs:
                            score = -self.evaluation_function(agent_obs, policy_agent)
                            total_score += score
                            score_details.append(f"{agent_id}→{team_goal}: {score:.2f}")

                        assignment[agent_id] = (team_id, team_goal)

                if not valid_combo:
                    continue

                ranked.append({
                    'score': total_score,
                    'assignment': assignment,
                    'partition': partition,
                    'goals': goal_combination,
                    'score_details': score_details
                })

        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked

    def compute_team_assignment_scores(self, observations_per_agent, possible_team_goals=None):
        """Compute scores for all possible team assignments."""
        agent_indices = list(range(self.num_agents))

        # Generate all possible team partitions
        all_partitions = self._generate_team_partitions(agent_indices, self.team_sizes)

        if possible_team_goals is None:
            possible_team_goals = self.generate_goal_combinations()

        return self._score_team_assignments(
            observations_per_agent,
            possible_team_goals,
            all_partitions
        )

    def compute_latency_metrics(
        self,
        observations_per_agent,
        possible_team_goals,
        true_assignment
    ) -> Dict[str, Any]:
        """Compute observation latency until correct predictions."""
        if possible_team_goals is None:
            possible_team_goals = self.generate_goal_combinations()

        per_agent_counts = {aid: len(obs) for aid, obs in observations_per_agent.items()}
        max_observations = max(per_agent_counts.values()) if per_agent_counts else 0
        team_partitions = self._generate_team_partitions(list(range(self.num_agents)), self.team_sizes)

        goal_latency = None
        team_latency = None
        joint_latency = None
        step_history = []

        if max_observations > 0:
            for step in range(1, max_observations + 1):
                truncated = {
                    aid: (obs if len(obs) <= step else obs[:step])
                    for aid, obs in observations_per_agent.items()
                }
                step_ranked = self._score_team_assignments(
                    truncated,
                    possible_team_goals,
                    team_partitions
                )

                if not step_ranked:
                    continue

                best_assignment = step_ranked[0]['assignment']
                goal_matches = team_matches = joint_matches = 0

                if true_assignment:
                    goal_matches = sum(
                        1 for aid in true_assignment.keys()
                        if best_assignment.get(aid, (None, None))[1] == true_assignment[aid][1]
                    )
                    team_matches = sum(
                        1 for aid in true_assignment.keys()
                        if best_assignment.get(aid, (None, None))[0] == true_assignment[aid][0]
                    )
                    joint_matches = sum(
                        1 for aid in true_assignment.keys()
                        if best_assignment.get(aid) == true_assignment[aid]
                    )

                    if goal_latency is None and goal_matches == self.num_agents:
                        goal_latency = step
                    if team_latency is None and team_matches == self.num_agents:
                        team_latency = step
                    if joint_latency is None and joint_matches == self.num_agents:
                        joint_latency = step

                step_history.append({
                    'step': step,
                    'best_assignment': best_assignment,
                    'best_score': step_ranked[0]['score'],
                    'goal_matches': goal_matches,
                    'team_matches': team_matches,
                    'joint_matches': joint_matches
                })

        return {
            'goal_latency': goal_latency,
            'team_latency': team_latency,
            'joint_latency': joint_latency,
            'max_observations': max_observations,
            'per_agent_counts': per_agent_counts,
            'step_history': step_history
        }

    def _generate_team_partitions(self, agents, team_sizes):
        """Generate all possible ways to partition agents into teams."""
        if len(team_sizes) == 0:
            return [[]]
        if len(team_sizes) == 1:
            return [[agents]]

        partitions = []
        first_team_size = team_sizes[0]
        remaining_sizes = team_sizes[1:]

        for first_team in combinations(agents, first_team_size):
            first_team = list(first_team)
            remaining_agents = [a for a in agents if a not in first_team]
            sub_partitions = self._generate_team_partitions(remaining_agents, remaining_sizes)

            for sub_partition in sub_partitions:
                partitions.append([first_team] + sub_partition)

        return partitions

    def display_multi_agent_grid(self, env, agent_positions, trajectories, step_num):
        """Display grid with multiple agents and their trajectories."""
        print(f"\n{'='*70}")
        print(f"STEP {step_num} - Environment State")
        print(f"{'='*70}")

        # Create grid - store both display and underlying info
        grid_display = [[' . ' for _ in range(env.size)] for _ in range(env.size)]
        grid_info = [['.' for _ in range(env.size)] for _ in range(env.size)]

        # Mark goals (lowest priority for display)
        for team_id, goal_pos in env.goal_positions.items():
            x, y = goal_pos
            grid_display[y][x] = f' G{team_id}'
            grid_info[y][x] = f'goal_{team_id}'

        # Mark trajectories (medium priority)
        for agent_id, trajectory in trajectories.items():
            for pos, _ in trajectory:
                x, y = int(pos[0]), int(pos[1])
                # Only mark if it's empty (don't overwrite goals yet)
                if grid_info[y][x] == '.':
                    grid_display[y][x] = ' * '
                    grid_info[y][x] = 'path'

        # Mark current agent positions (HIGHEST priority - always visible)
        team_symbols = ['A', 'B', 'C', 'D']
        position_counts = {}  # Track how many agents at each position

        for agent_id, pos in agent_positions.items():
            agent_idx = int(agent_id.split('_')[1])
            team_id = env.agent_teams[agent_idx]
            x, y = int(pos[0]), int(pos[1])
            pos_key = (x, y)

            # Count agents at this position
            if pos_key not in position_counts:
                position_counts[pos_key] = []
            position_counts[pos_key].append((agent_idx, team_id))

        # Display agents (show all if overlapping)
        for (x, y), agents_here in position_counts.items():
            if len(agents_here) == 1:
                agent_idx, team_id = agents_here[0]
                team_symbol = team_symbols[team_id % len(team_symbols)]
                grid_display[y][x] = f'{team_symbol}{agent_idx}'
                grid_info[y][x] = f'agent_{agent_idx}'
            else:
                # Multiple agents at same position - show count
                agent_list = ','.join([str(a[0]) for a in agents_here])
                grid_display[y][x] = f'[{len(agents_here)}]'
                grid_info[y][x] = f'agents_{agent_list}'

        # Print grid with better formatting
        print("\n    ", end="")
        for x in range(env.size):
            print(f"  {x} ", end="")
        print()
        print("    " + "----" * env.size)

        for y in range(env.size):
            print(f" {y} |", end="")
            for x in range(env.size):
                cell = grid_display[y][x]
                print(f" {cell}", end="")
            print(" |")
        print("    " + "----" * env.size)

        # Show agent positions clearly
        print("\nCurrent Agent Positions:")
        for agent_id in sorted(agent_positions.keys()):
            agent_idx = int(agent_id.split('_')[1])
            team_id = env.agent_teams[agent_idx]
            team_symbol = team_symbols[team_id % len(team_symbols)]
            pos = agent_positions[agent_id]
            print(f"  {team_symbol}{agent_idx} (Team {team_id}): ({int(pos[0])}, {int(pos[1])})", end="")

            # Show if at goal
            goal_pos = env.goal_positions[team_id]
            if np.array_equal(pos, goal_pos):
                print(" ✓ AT GOAL!", end="")
            print()

        print("\nLegend:")
        for team_id in range(env.num_teams):
            team_symbol = team_symbols[team_id % len(team_symbols)]
            goal = env.team_goals[team_id]
            goal_pos = env.goal_positions[team_id]
            agents_in_team = [i for i, t in enumerate(env.agent_teams) if t == team_id]
            print(f"  Team {team_id} ({team_symbol}): Agents {agents_in_team} → {goal} (goal at {goal_pos})")
        print(f"  G{team_id} = Goal markers")
        print(f"  *  = Path taken")

    def display_team_beliefs(
        self,
        best_assignments,
        true_assignment,
        observations_per_agent,
        possible_team_goals,
        top_k=5
    ):
        """Display top team assignment hypotheses."""
        print(f"\n{'='*60}")
        print("TOP TEAM ASSIGNMENT HYPOTHESES")
        print(f"{'='*60}")

        # Check if top prediction is correct
        top_assignment = best_assignments[0]['assignment']
        is_correct = all(
            top_assignment.get(aid, (None, None))[0] == true_assignment.get(aid, (None, None))[0]
            and top_assignment.get(aid, (None, None))[1] == true_assignment.get(aid, (None, None))[1]
            for aid in top_assignment.keys()
        )

        print(f"\nShowing top {min(top_k, len(best_assignments))} assignments:\n")

        for rank, assignment_info in enumerate(best_assignments[:top_k], 1):
            assignment = assignment_info['assignment']
            score = assignment_info['score']
            score_details = assignment_info.get('score_details', [])

            # Check if this matches true assignment
            is_true = all(
                assignment.get(aid, (None, None)) == true_assignment.get(aid, (None, None))
                for aid in assignment.keys()
            )

            marker = "✓" if is_true else (" " if rank == 1 else " ")
            rank_str = f"#{rank}"

            print(f"{marker} {rank_str:4s} Score: {score:8.2f}")

            # Group by teams
            teams_display = {}
            for agent_id, (team_id, goal) in assignment.items():
                if team_id not in teams_display:
                    teams_display[team_id] = []
                agent_num = agent_id.split('_')[1]
                teams_display[team_id].append((agent_num, goal))

            for team_id in sorted(teams_display.keys()):
                agents_goals = teams_display[team_id]
                agent_nums = [a for a, _ in agents_goals]
                goal = agents_goals[0][1]  # All agents in team have same goal
                print(f"      Team {team_id}: Agents {agent_nums} → {goal}")

            # Show score breakdown
            if score_details:
                print(f"      Breakdown: {', '.join(score_details)}")

            if is_true:
                print("      *** TRUE ASSIGNMENT ***")
            print()

        print(f"{'='*60}")

        if is_correct:
            print("\n✓ TOP PREDICTION IS CORRECT!")
        else:
            print("\n✗ Top prediction is incorrect")
            print("\nTrue assignment:")
            for agent_id, (team_id, goal) in true_assignment.items():
                print(f"  {agent_id}: Team {team_id} → {goal}")

        latency = self.compute_latency_metrics(
            observations_per_agent,
            possible_team_goals,
            true_assignment
        )
        max_obs = latency.get('max_observations', 0)
        if max_obs:
            goal_latency = latency.get('goal_latency')
            team_latency = latency.get('team_latency')
            joint_latency = latency.get('joint_latency')

            def fmt(value):
                return f"{value}/{max_obs}" if value is not None else f"not reached ≤ {max_obs}"

            print("\nLatency Summary:")
            print(f"  Goal lock-in: {fmt(goal_latency)}")
            print(f"  Team lock-in: {fmt(team_latency)}")
            print(f"  Joint lock-in: {fmt(joint_latency)}")

    def display_agent_observations(self, observations_per_agent, step_num):
        """Display observation counts for each agent."""
        print(f"\n{'='*60}")
        print(f"OBSERVATIONS COLLECTED (Step {step_num})")
        print(f"{'='*60}")

        for agent_id in sorted(observations_per_agent.keys()):
            obs_count = len(observations_per_agent[agent_id])
            print(f"  {agent_id}: {obs_count} observations")


def create_expert_policies(team_goals, agents, goal_names, grid_size=7):
    """Create expert policies using trained PPO models.

    Args:
        team_goals: List of team goal types (e.g., ['top_right', 'top_left'])
        agents: List of trained PPO agents
        goal_names: List of goal names corresponding to agents
        grid_size: Size of the grid
    """
    goal_positions = {}

    for goal_type in team_goals:
        if goal_type == 'top_right':
            goal_positions[goal_type] = np.array([grid_size - 1, 0])
        elif goal_type == 'top_left':
            goal_positions[goal_type] = np.array([0, 0])
        elif goal_type == 'bottom_left':
            goal_positions[goal_type] = np.array([0, grid_size - 1])
        elif goal_type == 'bottom_right':
            goal_positions[goal_type] = np.array([grid_size - 1, grid_size - 1])
        else:
            raise ValueError(f"Unsupported goal type: {goal_type}")

    def expert_policy(obs, goal_type, agent_idx):
        """Use trained PPO policy for the given goal.

        Args:
            obs: Flat observation array [agent0_x, agent0_y, agent1_x, agent1_y, ...]
            goal_type: Target goal type
            agent_idx: Index of the agent (to extract correct position from obs)
        """
        # Map goal_type to goal_name used in training
        goal_name_map = {
            'top_right': 'team_goal_top_right',
            'top_left': 'team_goal_top_left',
            'bottom_left': 'team_goal_bottom_left',
            'bottom_right': 'team_goal_bottom_right'
        }

        goal_name = goal_name_map.get(goal_type)
        if not goal_name:
            # Fallback to simple policy if goal not recognized
            return simple_policy(obs, goal_type, agent_idx)

        # Find the corresponding agent/model for this goal
        try:
            goal_idx = goal_names.index(goal_name)
            agent = agents[goal_idx]

            # Extract this agent's position from the full observation
            pos_offset = agent_idx * 2
            agent_pos = obs[pos_offset:pos_offset+2]

            # Get action probabilities from the trained model
            action_probs = agent.get_action_probabilities(agent_pos)

            # Use deterministic action (argmax) for consistency
            action = int(np.argmax(action_probs))

            return action
        except (ValueError, IndexError, AttributeError):
            # If there's any error, fall back to simple policy
            return simple_policy(obs, goal_type, agent_idx)

    def simple_policy(obs, goal_type, agent_idx):
        """Simple fallback policy that moves towards goal (doesn't handle obstacles well).

        Args:
            obs: Flat observation array [agent0_x, agent0_y, agent1_x, agent1_y, ...]
            goal_type: Target goal type
            agent_idx: Index of the agent (to extract correct position from obs)
        """
        goal_pos = goal_positions[goal_type]
        # Extract THIS agent's position from observation
        pos_offset = agent_idx * 2
        agent_pos = obs[pos_offset:pos_offset+2]

        diff = goal_pos - agent_pos
        if abs(diff[0]) > abs(diff[1]):
            return 3 if diff[0] > 0 else 2  # Right or Left
        else:
            return 0 if diff[1] < 0 else 1  # Up or Down

    return expert_policy


def main():
    print("="*80)
    print(" " * 10 + "MULTI-AGENT INCREMENTAL GOAL RECOGNITION")
    print("="*80)

    # Configuration
    EPISODES = 5000
    MODELS_DIR = "models"
    MAX_STEPS = 20

    # Check if models exist
    goal_names = [
        "team_goal_top_right",
        "team_goal_top_left",
        "team_goal_bottom_left",
        "team_goal_bottom_right"
    ]

    models_exist = all(
        os.path.exists(os.path.join(
            MODELS_DIR, f"episodes_{EPISODES}", goal, "PPOAgent", "model.pt"
        ))
        for goal in goal_names
    )

    if not models_exist:
        print("\n❌ Error: Trained models not found!")
        print(f"Please run 'python train_multi_agent.py --episodes {EPISODES}' first.")
        return

    # Load trained agents
    print("\nLoading trained agents...")
    env_classes = [
        TeamGoalTopRight,
        TeamGoalTopLeft,
        TeamGoalBottomLeft,
        TeamGoalBottomRight
    ]

    agents = []
    for env_class, goal_name in zip(env_classes, goal_names):
        agent = PPOAgent(
            env_name=env_class,
            models_dir=MODELS_DIR,
            goal_hypothesis=goal_name,
            episodes=EPISODES
        )
        agent.load_model()
        agents.append(agent)
        print(f"  ✓ Loaded {goal_name}")

    while True:
        print("\n" + "="*80)
        print("SELECT A SCENARIO")
        print("="*80)
        print("  1. Two teams, one agent each (Team A→top_right, Team B→top_left)")
        print("  2. Two teams, two agents each (Team A→top_right, Team B→top_left)")
        print("  3. Custom scenario")
        print("  0. Exit")

        try:
            choice = int(input("\nEnter your choice: "))
        except ValueError:
            print("Invalid input.")
            continue

        if choice == 0:
            break
        elif choice == 1:
            team_sizes = [1, 1]
            team_goals = ['top_right', 'top_left']
            num_agents = 2
        elif choice == 2:
            team_sizes = [2, 2]
            team_goals = ['top_right', 'top_left']
            num_agents = 4
        elif choice == 3:
            print("\nCustom scenario not yet implemented. Using scenario 1.")
            team_sizes = [1, 1]
            team_goals = ['top_right', 'top_left']
            num_agents = 2
        else:
            print("Invalid choice.")
            continue

        # Create environment
        env = MultiAgentGridWorld(
            size=7,
            max_steps=100,
            team_sizes=team_sizes,
            team_goals=team_goals,
            render_mode=None
        )

        # Create visualizer
        visualizer = MultiAgentIncrementalVisualizer(
            agents, goal_names, kl_divergence, num_agents, team_sizes
        )

        # Create expert policies using trained models
        expert_policy_fn = create_expert_policies(team_goals, agents, goal_names)

        # Generate trajectory
        print(f"\n{'='*80}")
        print(f"GENERATING TRAJECTORY")
        print(f"{'='*80}")
        print(f"Teams: {len(team_sizes)}")
        print(f"Team sizes: {team_sizes}")
        print(f"Team goals: {team_goals}")
        print(f"Total agents: {num_agents}")

        obs_dict, info = env.reset()
        initial_positions = info.get('initial_positions') or []
        if initial_positions:
            print("\nInitial agent positions (fixed):")
            for idx, pos in enumerate(initial_positions):
                print(f"  agent_{idx}: {tuple(pos)}")
        else:
            print("\nInitial agent positions: randomised (no preset available)")

        preset_pool = DEFAULT_INITIAL_POSITION_PRESETS.get(tuple(team_sizes), [])
        if preset_pool:
            if initial_positions:
                print(f"Preset {env.start_preset + 1}/{len(preset_pool)} selected.")
            else:
                print("Preset library available but not applied (random start).")

        # Define true assignment
        true_assignment = {}
        agent_counter = 0
        for team_id, team_size in enumerate(team_sizes):
            team_goal_str = f"team_goal_{team_goals[team_id]}"
            for _ in range(team_size):
                true_assignment[f'agent_{agent_counter}'] = (team_id, team_goal_str)
                agent_counter += 1

        # Select visualization mode
        print("\nVisualization modes:")
        print("  1. Step-by-step (press Enter after each step)")
        print("  2. Automatic with delay")

        try:
            mode = int(input("Select mode (1-2): "))
        except ValueError:
            mode = 1

        if mode == 2:
            try:
                delay = float(input("Enter delay between steps (seconds): "))
            except ValueError:
                delay = 1.0

        # Collect and visualize incrementally
        observations_per_agent = {f'agent_{i}': [] for i in range(num_agents)}
        trajectories = {f'agent_{i}': [] for i in range(num_agents)}
        # Extract each agent's position from the flat observation array [agent0_x, agent0_y, agent1_x, agent1_y, ...]
        current_positions = {}
        for i in range(num_agents):
            agent_id = f'agent_{i}'
            # Each agent's position is at offset i*2 in the observation array
            pos_offset = i * 2
            current_positions[agent_id] = obs_dict[agent_id][pos_offset:pos_offset+2].copy()

        # Possible goal combinations
        possible_team_goals = visualizer.generate_goal_combinations()

        for step in range(MAX_STEPS):
            if mode != 1:
                print("\033[2J\033[H")  # Clear screen

            print(f"\n{'='*80}")
            print(f"INCREMENTAL RECOGNITION - STEP {step + 1}/{MAX_STEPS}")
            print(f"{'='*80}")

            # Display grid
            visualizer.display_multi_agent_grid(env, current_positions, trajectories, step + 1)

            # Display observations collected so far
            visualizer.display_agent_observations(observations_per_agent, step + 1)

            # Compute and display team assignment beliefs (if we have observations)
            if step > 0:
                best_assignments = visualizer.compute_team_assignment_scores(
                    observations_per_agent, possible_team_goals
                )
                visualizer.display_team_beliefs(
                    best_assignments,
                    true_assignment,
                    observations_per_agent,
                    possible_team_goals,
                    top_k=3
                )
            else:
                print(f"\n{'='*60}")
                print("No observations yet - waiting for agent actions...")
                print(f"{'='*60}")

            # Take actions
            actions = {}
            for agent_id in observations_per_agent.keys():
                agent_idx = int(agent_id.split('_')[1])
                team_id = env.agent_teams[agent_idx]
                team_goal = team_goals[team_id]

                action = expert_policy_fn(obs_dict[agent_id], team_goal, agent_idx)
                actions[agent_id] = action

                # Store observation - extract this agent's position from the flat array
                pos_offset = agent_idx * 2
                agent_pos = obs_dict[agent_id][pos_offset:pos_offset+2].copy()
                observations_per_agent[agent_id].append((agent_pos, action))
                trajectories[agent_id].append((agent_pos, action))

            # Step environment
            obs_dict, rewards, terms, truncs, info = env.step(actions)

            # Update positions - extract each agent's position correctly
            for agent_id in current_positions.keys():
                agent_idx = int(agent_id.split('_')[1])
                pos_offset = agent_idx * 2
                current_positions[agent_id] = obs_dict[agent_id][pos_offset:pos_offset+2].copy()

            # Check if done
            if all(terms.values()) or all(truncs.values()):
                print(f"\nEpisode finished at step {step + 1}")
                print(f"Team success: {info['team_success']}")
                break

            # Control flow
            if mode == 1:
                input("\nPress Enter to continue...")
            else:
                time.sleep(delay)

        # Final summary
        print(f"\n{'='*80}")
        print("FINAL RECOGNITION RESULTS")
        print(f"{'='*80}")

        best_assignments = visualizer.compute_team_assignment_scores(
            observations_per_agent, possible_team_goals
        )

        top_assignment = best_assignments[0]['assignment']

        print("\nPredicted Assignment:")
        for agent_id, (team_id, goal) in sorted(top_assignment.items()):
            print(f"  {agent_id}: Team {team_id} → {goal}")

        print("\nTrue Assignment:")
        for agent_id, (team_id, goal) in sorted(true_assignment.items()):
            print(f"  {agent_id}: Team {team_id} → {goal}")

        # Check accuracy
        correct_teams = sum(
            1 for aid in top_assignment.keys()
            if top_assignment[aid][0] == true_assignment[aid][0]
        )
        correct_goals = sum(
            1 for aid in top_assignment.keys()
            if top_assignment[aid][1] == true_assignment[aid][1]
        )

        print(f"\nAccuracy:")
        print(f"  Team Assignment: {correct_teams}/{num_agents} ({correct_teams/num_agents:.1%})")
        print(f"  Goal Recognition: {correct_goals}/{num_agents} ({correct_goals/num_agents:.1%})")

        if correct_teams == num_agents and correct_goals == num_agents:
            print("\n✓ SUCCESS! All agents correctly assigned to teams and goals!")
        else:
            print("\n✗ Some agents incorrectly assigned.")

        input("\nPress Enter to return to menu...")

    print("\n" + "="*80)
    print("Thank you for using the multi-agent incremental recognition visualizer!")
    print("="*80)


if __name__ == "__main__":
    main()
