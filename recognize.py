#!/usr/bin/env python3
"""
Phase 2: Multi-Agent Goal and Team Recognition.

This script performs goal recognition and team assignment for multiple agents
using pre-trained policies. It supports different team configurations and
recognition methods.
"""

import argparse
import numpy as np
import sys
import os

# Add parent directory to path to access shared modules

from envs.multi_agent_grid_world import (
    MultiAgentGridWorld,
    TwoTeamsSingleAgent,
    TwoTeamsDoubleAgents
)
from envs.team_goal_environments import (
    TeamGoalTopRight,
    TeamGoalTopLeft,
    TeamGoalBottomLeft,
    TeamGoalBottomRight,
    TeamGoalCenter
)
from ml.ppo import PPOAgent
from recognizer.multi_agent_recognizer import MultiAgentRecognizer
from metrics.metrics import kl_divergence, cross_entropy, mean_action_distance
from typing import Dict, List, Tuple


def create_expert_policy(goal_type: str) -> callable:
    """Create an expert policy for a specific goal.

    Args:
        goal_type: Type of goal ('top_right', 'top_left', etc.)

    Returns:
        Policy function that maps observations to actions
    """
    def policy(obs):
        # Extract agent's own position from the full observation
        # Assuming obs contains all agent positions, get the first agent's position
        x, y = obs[0], obs[1]

        # Define goal positions
        goals = {
            'top_right': (6, 0),
            'top_left': (0, 0),
            'bottom_left': (0, 6),
            'bottom_right': (6, 6),
            'center': (3, 3)
        }

        goal_x, goal_y = goals.get(goal_type, (6, 0))

        # Simple expert policy: move towards goal
        if x < goal_x:
            return 3  # Right
        elif x > goal_x:
            return 2  # Left
        elif y < goal_y:
            return 1  # Down
        elif y > goal_y:
            return 0  # Up
        else:
            return np.random.randint(0, 4)  # Random if at goal

    return policy


def create_mixed_policy(goal1: str, goal2: str, switch_prob: float = 0.3) -> callable:
    """Create a mixed policy that sometimes pursues wrong goal (for testing).

    Args:
        goal1: Primary goal
        goal2: Secondary goal
        switch_prob: Probability of using secondary goal

    Returns:
        Mixed policy function
    """
    policy1 = create_expert_policy(goal1)
    policy2 = create_expert_policy(goal2)

    def mixed_policy(obs):
        if np.random.random() < switch_prob:
            return policy2(obs)
        return policy1(obs)

    return mixed_policy


def run_scenario_1(recognizer, args):
    """Scenario 1: Two teams with one agent each."""
    print("\n" + "="*70)
    print("SCENARIO 1: Two Teams, One Agent Each")
    print("="*70)
    print("Team A (Agent 0) → Goal: Top Right")
    print("Team B (Agent 1) → Goal: Top Left")

    # Create environment
    env = TwoTeamsSingleAgent(render_mode=None)

    # Define true assignment
    true_assignment = {
        'agent_0': (0, 'team_goal_top_right'),
        'agent_1': (1, 'team_goal_top_left')
    }

    # Create expert policies for data collection
    policies = {
        'agent_0': create_expert_policy('top_right'),
        'agent_1': create_expert_policy('top_left')
    }

    # Collect observations
    print("\nCollecting observations...")
    observations = recognizer.collect_multi_agent_observations(
        env, num_steps=args.observation_steps, policies=policies
    )

    # Define possible goal combinations for teams
    # In this case, we know there are 2 teams, each can have different goals
    possible_goals = [
        ('team_goal_top_right', 'team_goal_top_left'),    # Correct assignment
        ('team_goal_top_left', 'team_goal_top_right'),    # Swapped
        ('team_goal_bottom_left', 'team_goal_bottom_right'),
        ('team_goal_center', 'team_goal_top_left'),
        # Add more combinations as needed
    ]

    # Perform recognition
    print("\nPerforming joint team-goal recognition...")
    result = recognizer.recognize_with_team_assignment(
        observations, possible_goals, true_assignment
    )

    return result


def run_scenario_2(recognizer, args):
    """Scenario 2: Two teams with two agents each."""
    print("\n" + "="*70)
    print("SCENARIO 2: Two Teams, Two Agents Each")
    print("="*70)
    print("Team A (Agents 0, 1) → Goal: Top Right")
    print("Team B (Agents 2, 3) → Goal: Top Left")

    # Update recognizer for 4 agents
    recognizer.num_agents = 4
    recognizer.team_sizes = [2, 2]

    # Create environment
    env = TwoTeamsDoubleAgents(render_mode=None)

    # Define true assignment
    true_assignment = {
        'agent_0': (0, 'team_goal_top_right'),
        'agent_1': (0, 'team_goal_top_right'),
        'agent_2': (1, 'team_goal_top_left'),
        'agent_3': (1, 'team_goal_top_left')
    }

    # Create expert policies
    policies = {
        'agent_0': create_expert_policy('top_right'),
        'agent_1': create_expert_policy('top_right'),
        'agent_2': create_expert_policy('top_left'),
        'agent_3': create_expert_policy('top_left')
    }

    # Collect observations
    print("\nCollecting observations...")
    observations = recognizer.collect_multi_agent_observations(
        env, num_steps=args.observation_steps, policies=policies
    )

    # Possible goal combinations
    possible_goals = [
        ('team_goal_top_right', 'team_goal_top_left'),    # Correct
        ('team_goal_top_left', 'team_goal_top_right'),    # Swapped
        ('team_goal_bottom_left', 'team_goal_bottom_right'),
        ('team_goal_center', 'team_goal_center'),
    ]

    # Perform recognition
    print("\nPerforming joint team-goal recognition...")
    result = recognizer.recognize_with_team_assignment(
        observations, possible_goals, true_assignment
    )

    return result


def run_scenario_3(recognizer, args):
    """Scenario 3: Mixed teams with noisy observations."""
    print("\n" + "="*70)
    print("SCENARIO 3: Mixed Teams with Noisy Policies")
    print("="*70)
    print("Team A (Agent 0) → Goal: Top Right (with 30% noise)")
    print("Team B (Agent 1) → Goal: Top Left (with 30% noise)")

    # Reset to 2 agents
    recognizer.num_agents = 2
    recognizer.team_sizes = [1, 1]

    # Create environment
    env = TwoTeamsSingleAgent(render_mode=None)

    # Define true assignment
    true_assignment = {
        'agent_0': (0, 'team_goal_top_right'),
        'agent_1': (1, 'team_goal_top_left')
    }

    # Create noisy policies (sometimes pursue wrong goal)
    policies = {
        'agent_0': create_mixed_policy('top_right', 'bottom_left', 0.3),
        'agent_1': create_mixed_policy('top_left', 'bottom_right', 0.3)
    }

    # Collect observations
    print("\nCollecting observations with noisy policies...")
    observations = recognizer.collect_multi_agent_observations(
        env, num_steps=args.observation_steps, policies=policies
    )

    # Possible goal combinations
    possible_goals = [
        ('team_goal_top_right', 'team_goal_top_left'),
        ('team_goal_top_left', 'team_goal_top_right'),
        ('team_goal_bottom_left', 'team_goal_bottom_right'),
        ('team_goal_bottom_right', 'team_goal_bottom_left'),
    ]

    # Perform recognition
    print("\nPerforming recognition with noisy observations...")
    result = recognizer.recognize_with_team_assignment(
        observations, possible_goals, true_assignment
    )

    return result


def main():
    parser = argparse.ArgumentParser(description='Multi-agent goal and team recognition')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of episodes used for training')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory with trained models')
    parser.add_argument('--observation-steps', type=int, default=30,
                        help='Number of observation steps to collect')
    parser.add_argument('--metric', type=str, default='kl_divergence',
                        choices=['kl_divergence', 'cross_entropy', 'mean_action_distance'],
                        help='Evaluation metric to use')
    parser.add_argument('--scenario', type=int, default=0,
                        help='Scenario to run (0=all, 1-3=specific)')
    args = parser.parse_args()

    print("="*70)
    print("MULTI-AGENT GOAL AND TEAM RECOGNITION")
    print("="*70)
    print(f"Using models from: {args.models_dir}/episodes_{args.episodes}/")
    print(f"Observation steps: {args.observation_steps}")
    print(f"Evaluation metric: {args.metric}")
    print("="*70)

    # Select evaluation metric
    metrics = {
        'kl_divergence': kl_divergence,
        'cross_entropy': cross_entropy,
        'mean_action_distance': mean_action_distance
    }
    eval_function = metrics[args.metric]

    # Define team goal environments
    team_goal_classes = [
        TeamGoalTopRight,
        TeamGoalTopLeft,
        TeamGoalBottomLeft,
        TeamGoalBottomRight,
        TeamGoalCenter
    ]

    goal_names = [
        "team_goal_top_right",
        "team_goal_top_left",
        "team_goal_bottom_left",
        "team_goal_bottom_right",
        "team_goal_center"
    ]

    # Create recognizer with pre-trained models
    print("\nLoading pre-trained models...")
    recognizer = MultiAgentRecognizer(
        agent_class=PPOAgent,
        goal_env_classes=team_goal_classes,
        models_dir=args.models_dir,
        goal_names=goal_names,
        evaluation_function=eval_function,
        num_agents=2,  # Default, will be updated per scenario
        team_sizes=[1, 1],  # Default, will be updated per scenario
        episodes=args.episodes,
        train_new=False  # Load existing models
    )

    # Run scenarios
    results = []

    if args.scenario == 0 or args.scenario == 1:
        result1 = run_scenario_1(recognizer, args)
        results.append(('Scenario 1', result1))

    if args.scenario == 0 or args.scenario == 2:
        result2 = run_scenario_2(recognizer, args)
        results.append(('Scenario 2', result2))

    if args.scenario == 0 or args.scenario == 3:
        result3 = run_scenario_3(recognizer, args)
        results.append(('Scenario 3', result3))

    # Summary
    print("\n" + "="*70)
    print("RECOGNITION SUMMARY")
    print("="*70)

    for scenario_name, result in results:
        print(f"\n{scenario_name}:")
        if 'accuracy' in result and result['accuracy']:
            print(f"  Goal Accuracy: {result['accuracy']['goal_accuracy']:.2%}")
            print(f"  Team Accuracy: {result['accuracy']['team_accuracy']:.2%}")
        else:
            print("  No ground truth available")

    print("\n" + "="*70)
    print("Recognition complete!")
    print("="*70)


if __name__ == "__main__":
    main()