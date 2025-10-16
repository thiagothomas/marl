"""Multi-agent goal and team recognizer implementation."""

import os
import numpy as np
from typing import List, Tuple, Optional, Callable, Type, Dict, Any
from itertools import product, permutations
from ml.base_agent import RLAgent
from ml.ppo import PPOAgent
from metrics.metrics import softmin
import gymnasium as gym
from .recognizer import Recognizer


class MultiAgentRecognizer(Recognizer):
    """Goal and team recognizer for multi-agent scenarios.

    This recognizer extends the single-agent recognizer to handle:
    - Multiple agents acting simultaneously
    - Team assignment (which agents belong to which team)
    - Goal recognition (what goal each team is pursuing)
    """

    def __init__(
        self,
        agent_class: Type[RLAgent],
        goal_env_classes: List,
        models_dir: str,
        goal_names: List[str],
        evaluation_function: Callable,
        num_agents: int,
        team_sizes: List[int],
        episodes: int = 10000,
        train_new: bool = True
    ):
        """Initialize the multi-agent goal recognizer.

        Args:
            agent_class: Class of RL agent to use (e.g., PPOAgent)
            goal_env_classes: List of environment classes for each possible goal
            models_dir: Directory to save/load models
            goal_names: Names of goals corresponding to environments
            evaluation_function: Function to evaluate observation match
            num_agents: Total number of agents
            team_sizes: List of team sizes [team1_size, team2_size, ...]
            episodes: Number of training episodes
            train_new: Whether to train new agents or load existing ones
        """
        # Initialize parent class for single-agent policy training/loading
        super().__init__(
            agent_class=agent_class,
            env_classes=goal_env_classes,
            models_dir=models_dir,
            goal_names=goal_names,
            evaluation_function=evaluation_function,
            episodes=episodes,
            train_new=train_new
        )

        self.num_agents = num_agents
        self.team_sizes = team_sizes
        self.num_teams = len(team_sizes)

        # Validate inputs
        assert sum(team_sizes) == num_agents, \
            f"Sum of team sizes {sum(team_sizes)} doesn't match num_agents {num_agents}"

    def collect_multi_agent_observations(
        self,
        env: gym.Env,
        num_steps: int = 50,
        policies: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, List[Tuple[np.ndarray, int]]]:
        """Collect observations from multiple agents acting in the environment.

        Args:
            env: Multi-agent environment
            num_steps: Number of steps to collect
            policies: Optional dict of policies per agent (if None, uses random)

        Returns:
            Dict mapping agent_id to list of (observation, action) tuples
        """
        observations_per_agent = {
            f'agent_{i}': [] for i in range(self.num_agents)
        }

        obs_dict, _ = env.reset()

        for step in range(num_steps):
            actions = {}

            for agent_id in observations_per_agent.keys():
                if policies and agent_id in policies:
                    # Use provided policy
                    action = policies[agent_id](obs_dict[agent_id])
                else:
                    # Random policy
                    action = env.action_space[agent_id].sample()

                actions[agent_id] = action

                # Store observation-action pair
                # Note: We store individual agent's view of the state
                # In our case, all agents see the full state, but we extract
                # their position for individual behavior analysis
                agent_idx = int(agent_id.split('_')[1])
                agent_pos = obs_dict[agent_id][agent_idx*2:(agent_idx+1)*2].copy()
                observations_per_agent[agent_id].append((agent_pos, action))

            # Execute all actions
            obs_dict, rewards, terms, truncs, info = env.step(actions)

            # Check if episode is done
            all_done = all(terms.values()) or all(truncs.values())
            if all_done:
                obs_dict, _ = env.reset()

        return observations_per_agent

    def recognize_independent(
        self,
        observations_per_agent: Dict[str, List[Tuple[np.ndarray, int]]],
        real_goals_per_agent: Optional[Dict[str, str]] = None,
        real_teams_per_agent: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Recognize goals for each agent independently.

        This approach treats each agent independently and finds the best
        goal match for each one without considering team constraints.

        Args:
            observations_per_agent: Dict of observations per agent
            real_goals_per_agent: Ground truth goals per agent (for evaluation)
            real_teams_per_agent: Ground truth teams per agent (for evaluation)

        Returns:
            Recognition results
        """
        results_per_agent = {}

        print("\n" + "="*60)
        print("INDEPENDENT AGENT RECOGNITION")
        print("="*60)

        for agent_id, agent_obs in observations_per_agent.items():
            scores = []

            # Evaluate against each possible goal
            for i, (agent, goal_name) in enumerate(zip(self.agents, self.goal_names)):
                score = self.evaluation_function(agent_obs, agent)
                scores.append(score)

            scores = np.array(scores)
            probabilities = softmin(scores)

            # Get best goal
            best_goal_idx = np.argmax(probabilities)
            best_goal = self.goal_names[best_goal_idx]

            results_per_agent[agent_id] = {
                'predicted_goal': best_goal,
                'goal_probabilities': probabilities,
                'scores': scores
            }

            print(f"\n{agent_id}:")
            print(f"  Predicted goal: {best_goal} (prob: {probabilities[best_goal_idx]:.3f})")

        # Check accuracy if ground truth provided
        if real_goals_per_agent:
            correct = sum(
                1 for agent_id, result in results_per_agent.items()
                if result['predicted_goal'] == real_goals_per_agent[agent_id]
            )
            accuracy = correct / len(results_per_agent)
            print(f"\nGoal Recognition Accuracy: {accuracy:.2%} ({correct}/{len(results_per_agent)})")

        return {
            'agent_results': results_per_agent,
            'method': 'independent'
        }

    def recognize_with_team_assignment(
        self,
        observations_per_agent: Dict[str, List[Tuple[np.ndarray, int]]],
        possible_team_goals: List[Tuple[str, ...]],
        real_assignment: Optional[Dict[str, Tuple[int, str]]] = None
    ) -> Dict[str, Any]:
        """Recognize goals and team assignments jointly.

        This approach considers all possible team-goal assignments and finds
        the best overall match using brute force search.

        Args:
            observations_per_agent: Dict of observations per agent
            possible_team_goals: List of possible goal combinations for teams
                                Example: [('top_right', 'top_left'), ('top_left', 'top_right')]
            real_assignment: Ground truth (team_id, goal) per agent

        Returns:
            Recognition results with team assignments
        """
        print("\n" + "="*60)
        print("JOINT TEAM-GOAL RECOGNITION (BRUTE FORCE)")
        print("="*60)

        agent_ids = list(observations_per_agent.keys())
        best_score = float('-inf')
        best_assignment = None

        # Generate all possible ways to assign agents to teams
        # This is a partition problem: divide N agents into teams of specific sizes
        agent_indices = list(range(self.num_agents))

        # Generate all possible team assignments
        all_team_assignments = self._generate_team_partitions(agent_indices, self.team_sizes)

        print(f"Evaluating {len(all_team_assignments)} team assignments")
        print(f"with {len(possible_team_goals)} goal combinations each")
        print(f"Total combinations: {len(all_team_assignments) * len(possible_team_goals)}")

        # Try all combinations of team assignments and goal assignments
        for team_partition in all_team_assignments:
            for goal_combination in possible_team_goals:
                total_score = 0

                # Build assignment dict
                current_assignment = {}
                for team_id, team_agents in enumerate(team_partition):
                    team_goal = goal_combination[team_id]
                    goal_idx = self.goal_names.index(team_goal)

                    for agent_idx in team_agents:
                        agent_id = f'agent_{agent_idx}'
                        agent_obs = observations_per_agent[agent_id]

                        # Evaluate this agent against this goal
                        score = -self.evaluation_function(agent_obs, self.agents[goal_idx])
                        total_score += score

                        current_assignment[agent_id] = (team_id, team_goal)

                if total_score > best_score:
                    best_score = total_score
                    best_assignment = current_assignment

        # Display results
        print("\n" + "-"*60)
        print("BEST ASSIGNMENT FOUND:")
        print("-"*60)

        team_assignments = {}
        for team_id in range(self.num_teams):
            team_agents = [
                agent_id for agent_id, (t_id, _) in best_assignment.items()
                if t_id == team_id
            ]
            team_goal = best_assignment[team_agents[0]][1] if team_agents else None
            team_assignments[f'Team_{team_id}'] = {
                'agents': team_agents,
                'goal': team_goal
            }
            print(f"Team {team_id}: {team_agents} → Goal: {team_goal}")

        # Check accuracy if ground truth provided
        accuracy_info = {}
        if real_assignment:
            correct_goals = sum(
                1 for agent_id, (_, goal) in best_assignment.items()
                if goal == real_assignment[agent_id][1]
            )
            correct_teams = sum(
                1 for agent_id, (team, _) in best_assignment.items()
                if team == real_assignment[agent_id][0]
            )

            goal_accuracy = correct_goals / self.num_agents
            team_accuracy = correct_teams / self.num_agents

            print(f"\nAccuracy:")
            print(f"  Goal Recognition: {goal_accuracy:.2%} ({correct_goals}/{self.num_agents})")
            print(f"  Team Assignment: {team_accuracy:.2%} ({correct_teams}/{self.num_agents})")

            accuracy_info = {
                'goal_accuracy': goal_accuracy,
                'team_accuracy': team_accuracy,
                'correct_goals': correct_goals,
                'correct_teams': correct_teams
            }

        return {
            'best_assignment': best_assignment,
            'team_assignments': team_assignments,
            'best_score': best_score,
            'method': 'joint_brute_force',
            'accuracy': accuracy_info
        }

    def _generate_team_partitions(self, agents: List[int], team_sizes: List[int]) -> List[List[List[int]]]:
        """Generate all possible ways to partition agents into teams.

        Args:
            agents: List of agent indices
            team_sizes: List of team sizes

        Returns:
            List of partitions, where each partition is a list of teams
        """
        if len(team_sizes) == 0:
            return [[]]

        if len(team_sizes) == 1:
            return [[agents]]

        partitions = []
        first_team_size = team_sizes[0]
        remaining_sizes = team_sizes[1:]

        # Generate all combinations for the first team
        from itertools import combinations
        for first_team in combinations(agents, first_team_size):
            first_team = list(first_team)
            remaining_agents = [a for a in agents if a not in first_team]

            # Recursively partition the remaining agents
            sub_partitions = self._generate_team_partitions(remaining_agents, remaining_sizes)

            for sub_partition in sub_partitions:
                partitions.append([first_team] + sub_partition)

        return partitions

    def evaluate_on_multi_agent_scenarios(
        self,
        test_scenarios: List[Dict[str, Any]],
        method: str = 'joint'
    ) -> Dict[str, Any]:
        """Evaluate recognition on multiple multi-agent test scenarios.

        Args:
            test_scenarios: List of test scenarios, each containing:
                - 'env': Multi-agent environment
                - 'true_assignment': Ground truth (team, goal) per agent
                - 'possible_goals': Possible goal combinations
            method: Recognition method ('independent' or 'joint')

        Returns:
            Evaluation results
        """
        print("\n" + "="*70)
        print("MULTI-AGENT EVALUATION")
        print("="*70)

        all_results = []
        total_goal_correct = 0
        total_team_correct = 0
        total_agents = 0

        for i, scenario in enumerate(test_scenarios):
            print(f"\n--- Scenario {i+1}/{len(test_scenarios)} ---")

            env = scenario['env']
            true_assignment = scenario['true_assignment']
            possible_goals = scenario.get('possible_goals', [tuple(self.goal_names)])

            # Collect observations
            observations = self.collect_multi_agent_observations(env, num_steps=50)

            # Recognize based on method
            if method == 'joint':
                result = self.recognize_with_team_assignment(
                    observations, possible_goals, true_assignment
                )
                if 'accuracy' in result and result['accuracy']:
                    total_goal_correct += result['accuracy']['correct_goals']
                    total_team_correct += result['accuracy']['correct_teams']
                    total_agents += self.num_agents
            else:
                # For independent method
                true_goals = {aid: assign[1] for aid, assign in true_assignment.items()}
                true_teams = {aid: assign[0] for aid, assign in true_assignment.items()}
                result = self.recognize_independent(
                    observations, true_goals, true_teams
                )

            all_results.append(result)

        # Overall accuracy
        if total_agents > 0:
            overall_goal_accuracy = total_goal_correct / total_agents
            overall_team_accuracy = total_team_correct / total_agents

            print("\n" + "="*70)
            print("OVERALL RESULTS:")
            print("="*70)
            print(f"Goal Recognition Accuracy: {overall_goal_accuracy:.2%}")
            print(f"Team Assignment Accuracy: {overall_team_accuracy:.2%}")

        return {
            'all_results': all_results,
            'overall_goal_accuracy': overall_goal_accuracy if total_agents > 0 else None,
            'overall_team_accuracy': overall_team_accuracy if total_agents > 0 else None
        }