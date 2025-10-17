"""Multi-agent goal and team recognizer implementation."""

import os
import numpy as np
from typing import List, Tuple, Optional, Callable, Type, Dict, Any
from itertools import permutations
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
        policies: Optional[Dict[str, Callable]] = None,
        initial_obs: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, List[Tuple[np.ndarray, int]]]:
        """Collect observations from multiple agents acting in the environment.

        Args:
            env: Multi-agent environment
            num_steps: Number of steps to collect
            policies: Optional dict of policies per agent (if None, uses random)
            initial_obs: Optional observation dict to start from (skips env.reset())

        Returns:
            Dict mapping agent_id to list of (observation, action) tuples
        """
        observations_per_agent = {
            f'agent_{i}': [] for i in range(self.num_agents)
        }

        if initial_obs is None:
            obs_dict, _ = env.reset()
        else:
            obs_dict = {
                agent_id: np.array(obs, copy=True)
                for agent_id, obs in initial_obs.items()
            }

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

    def _score_team_assignments(
        self,
        observations_per_agent: Dict[str, List[Tuple[np.ndarray, int]]],
        possible_team_goals: List[Tuple[str, ...]],
        team_partitions: List[List[List[int]]]
    ) -> List[Dict[str, Any]]:
        """Score all team-goal assignment combinations."""
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
                            score_details.append(f"{agent_id}→{team_goal}: {score:.4f}")

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

        ranked.sort(key=lambda item: item['score'], reverse=True)
        return ranked

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

        def truncate_observations(step: int) -> Dict[str, List[Tuple[np.ndarray, int]]]:
            truncated: Dict[str, List[Tuple[np.ndarray, int]]] = {}
            for agent_id, obs_list in observations_per_agent.items():
                if len(obs_list) <= step:
                    truncated[agent_id] = obs_list.copy()
                else:
                    truncated[agent_id] = obs_list[:step]
            return truncated

        # Generate all possible ways to assign agents to teams
        # This is a partition problem: divide N agents into teams of specific sizes
        agent_indices = list(range(self.num_agents))

        # Generate all possible team assignments
        all_team_assignments = self._generate_team_partitions(agent_indices, self.team_sizes)

        print(f"Evaluating {len(all_team_assignments)} team assignments")
        print(f"with {len(possible_team_goals)} goal combinations each")
        print(f"Total combinations: {len(all_team_assignments) * len(possible_team_goals)}")

        ranked_assignments = self._score_team_assignments(
            observations_per_agent,
            possible_team_goals,
            all_team_assignments
        )

        if not ranked_assignments:
            raise ValueError("No valid team-goal assignments produced a score. Check possible_team_goals.")

        best_assignment = ranked_assignments[0]['assignment']
        best_score = ranked_assignments[0]['score']

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

        # Recognition latency (observations needed for correctness)
        per_agent_counts = {aid: len(obs) for aid, obs in observations_per_agent.items()}
        max_observations = max(per_agent_counts.values()) if per_agent_counts else 0
        step_history: List[Dict[str, Any]] = []
        goal_latency = None
        team_latency = None
        joint_latency = None

        if max_observations > 0:
            for step in range(1, max_observations + 1):
                truncated_obs = truncate_observations(step)
                step_rankings = self._score_team_assignments(
                    truncated_obs,
                    possible_team_goals,
                    all_team_assignments
                )

                if not step_rankings:
                    continue

                step_best = step_rankings[0]['assignment']
                goal_matches = team_matches = joint_matches = 0

                if real_assignment:
                    goal_matches = sum(
                        1 for aid in real_assignment.keys()
                        if step_best.get(aid, (None, None))[1] == real_assignment[aid][1]
                    )
                    team_matches = sum(
                        1 for aid in real_assignment.keys()
                        if step_best.get(aid, (None, None))[0] == real_assignment[aid][0]
                    )
                    joint_matches = sum(
                        1 for aid in real_assignment.keys()
                        if step_best.get(aid) == real_assignment[aid]
                    )

                    if goal_latency is None and goal_matches == self.num_agents:
                        goal_latency = step
                    if team_latency is None and team_matches == self.num_agents:
                        team_latency = step
                    if joint_latency is None and joint_matches == self.num_agents:
                        joint_latency = step

                step_history.append({
                    'step': step,
                    'best_assignment': step_best,
                    'best_score': step_rankings[0]['score'],
                    'goal_matches': goal_matches,
                    'team_matches': team_matches,
                    'joint_matches': joint_matches,
                    'top_assignments': step_rankings[:3]
                })

        if real_assignment and max_observations > 0:
            print("\nRecognition Latency:")
            goal_msg = (
                f"  Goals correct at observation {goal_latency}/{max_observations}"
                if goal_latency is not None else
                f"  Goals never fully correct within {max_observations} observations"
            )
            team_msg = (
                f"  Teams correct at observation {team_latency}/{max_observations}"
                if team_latency is not None else
                f"  Teams never fully correct within {max_observations} observations"
            )
            joint_msg = (
                f"  Joint assignment correct at observation {joint_latency}/{max_observations}"
                if joint_latency is not None else
                f"  Joint assignment never fully correct within {max_observations} observations"
            )
            print(goal_msg)
            print(team_msg)
            print(joint_msg)

        return {
            'best_assignment': best_assignment,
            'team_assignments': team_assignments,
            'best_score': best_score,
            'method': 'joint_brute_force',
            'accuracy': accuracy_info,
            'ranked_assignments': ranked_assignments,
            'latency': {
                'goal_latency': goal_latency,
                'team_latency': team_latency,
                'joint_latency': joint_latency,
                'max_observations': max_observations,
                'per_agent_counts': per_agent_counts,
                'step_history': step_history
            }
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
