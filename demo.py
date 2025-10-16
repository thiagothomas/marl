#!/usr/bin/env python3
"""
Complete Multi-Agent Demo: Training and Recognition.

This script demonstrates the full multi-agent goal recognition pipeline:
1. Train policies for team goals (if needed)
2. Run multi-agent scenarios
3. Perform team assignment and goal recognition
4. Visualize results
"""

import os
import argparse
import numpy as np
import sys

# Add parent directory to path to access shared modules

from envs.multi_agent_grid_world import MultiAgentGridWorld
from envs.team_goal_environments import (
    TeamGoalTopRight,
    TeamGoalTopLeft,
    TeamGoalBottomLeft,
    TeamGoalBottomRight,
    TeamGoalCenter
)
from ml.ppo import PPOAgent
from recognizer.multi_agent_recognizer import MultiAgentRecognizer
from metrics.metrics import kl_divergence


def visualize_scenario(env, policies, steps=20):
    """Visualize agents moving in the environment."""
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)

    obs_dict, _ = env.reset()
    env.render()

    for step in range(steps):
        print(f"\nStep {step + 1}/{steps}")

        actions = {}
        for agent_id in obs_dict.keys():
            if agent_id in policies:
                actions[agent_id] = policies[agent_id](obs_dict[agent_id])
            else:
                actions[agent_id] = env.action_space[agent_id].sample()

        obs_dict, rewards, terms, truncs, info = env.step(actions)
        env.render()

        # Show rewards
        print("Rewards:", {k: f"{v:.2f}" for k, v in rewards.items()})

        # Check if done
        if all(terms.values()) or all(truncs.values()):
            print("\nEpisode finished!")
            print("Team success:", info['team_success'])
            break

    return info


def demo_scenario_basic():
    """Basic demo: 2 teams, 1 agent each."""
    print("\n" + "="*70)
    print("DEMO: Basic Scenario (2 teams, 1 agent each)")
    print("="*70)

    # Create environment
    env = MultiAgentGridWorld(
        size=7,
        max_steps=100,
        team_sizes=[1, 1],
        team_goals=['top_right', 'top_left'],
        render_mode='console'
    )

    print("\nScenario Setup:")
    print("- Team 0 (Agent 0): Goal = reach top-right corner")
    print("- Team 1 (Agent 1): Goal = reach top-left corner")

    # Create simple expert policies
    def policy_top_right(obs):
        x, y = obs[0], obs[1]  # Get agent 0's position
        if x < 6:
            return 3  # Right
        elif y > 0:
            return 0  # Up
        else:
            return np.random.randint(0, 4)

    def policy_top_left(obs):
        x, y = obs[2], obs[3]  # Get agent 1's position
        if x > 0:
            return 2  # Left
        elif y > 0:
            return 0  # Up
        else:
            return np.random.randint(0, 4)

    policies = {
        'agent_0': policy_top_right,
        'agent_1': policy_top_left
    }

    # Run and visualize
    info = visualize_scenario(env, policies, steps=20)

    return env, policies


def demo_scenario_complex():
    """Complex demo: 2 teams, 2 agents each."""
    print("\n" + "="*70)
    print("DEMO: Complex Scenario (2 teams, 2 agents each)")
    print("="*70)

    # Create environment
    env = MultiAgentGridWorld(
        size=7,
        max_steps=100,
        team_sizes=[2, 2],
        team_goals=['bottom_right', 'top_left'],
        render_mode='console'
    )

    print("\nScenario Setup:")
    print("- Team 0 (Agents 0, 1): Goal = reach bottom-right corner")
    print("- Team 1 (Agents 2, 3): Goal = reach top-left corner")

    # Create policies
    def policy_bottom_right(agent_idx):
        def policy(obs):
            x, y = obs[agent_idx*2], obs[agent_idx*2 + 1]
            if x < 6:
                return 3  # Right
            elif y < 6:
                return 1  # Down
            else:
                return np.random.randint(0, 4)
        return policy

    def policy_top_left(agent_idx):
        def policy(obs):
            x, y = obs[agent_idx*2], obs[agent_idx*2 + 1]
            if x > 0:
                return 2  # Left
            elif y > 0:
                return 0  # Up
            else:
                return np.random.randint(0, 4)
        return policy

    policies = {
        'agent_0': policy_bottom_right(0),
        'agent_1': policy_bottom_right(1),
        'agent_2': policy_top_left(2),
        'agent_3': policy_top_left(3)
    }

    # Run and visualize
    info = visualize_scenario(env, policies, steps=30)

    return env, policies


def run_recognition_demo(episodes=1000):
    """Run complete recognition demo with pre-trained or newly trained models."""
    print("\n" + "="*70)
    print("MULTI-AGENT RECOGNITION DEMO")
    print("="*70)

    # Check if models exist
    models_dir = 'models'
    model_path = os.path.join(models_dir, f'episodes_{episodes}', 'team_goal_top_right', 'PPOAgent', 'model.pt')

    if not os.path.exists(model_path):
        print(f"\nModels not found at {models_dir}/episodes_{episodes}/")
        print("Training new models... (this will take a few minutes)")

        # Train models
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

        recognizer = MultiAgentRecognizer(
            agent_class=PPOAgent,
            goal_env_classes=team_goal_classes,
            models_dir=models_dir,
            goal_names=goal_names,
            evaluation_function=kl_divergence,
            num_agents=2,
            team_sizes=[1, 1],
            episodes=episodes,
            train_new=True
        )
        print("\nTraining complete!")
    else:
        print(f"\nUsing existing models from {models_dir}/episodes_{episodes}/")

    # Now run recognition demo
    print("\n" + "="*70)
    print("RECOGNITION PHASE")
    print("="*70)

    # Load trained models
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

    recognizer = MultiAgentRecognizer(
        agent_class=PPOAgent,
        goal_env_classes=team_goal_classes,
        models_dir=models_dir,
        goal_names=goal_names,
        evaluation_function=kl_divergence,
        num_agents=2,
        team_sizes=[1, 1],
        episodes=episodes,
        train_new=False
    )

    # Test case: Unknown team assignments
    print("\nTest Case: Observe agents and determine their teams/goals")
    print("-" * 60)

    # Create a test environment
    env = MultiAgentGridWorld(
        size=7,
        max_steps=100,
        team_sizes=[1, 1],
        team_goals=['bottom_right', 'top_left'],
        render_mode=None
    )

    # Create test policies (hidden from recognizer)
    def policy_br(obs):
        x, y = obs[0], obs[1]
        if x < 6: return 3
        elif y < 6: return 1
        else: return 0

    def policy_tl(obs):
        x, y = obs[2], obs[3]
        if x > 0: return 2
        elif y > 0: return 0
        else: return 1

    test_policies = {
        'agent_0': policy_br,  # Going bottom-right
        'agent_1': policy_tl   # Going top-left
    }

    # Collect observations
    print("Collecting observations from unknown agents...")
    observations = recognizer.collect_multi_agent_observations(
        env, num_steps=30, policies=test_policies
    )

    # Define possible team-goal combinations
    possible_goals = [
        ('team_goal_top_right', 'team_goal_top_left'),
        ('team_goal_top_left', 'team_goal_top_right'),
        ('team_goal_bottom_right', 'team_goal_top_left'),
        ('team_goal_bottom_left', 'team_goal_bottom_right'),
        ('team_goal_center', 'team_goal_center'),
    ]

    # Perform recognition
    print("\nAnalyzing behaviors to determine teams and goals...")
    true_assignment = {
        'agent_0': (0, 'team_goal_bottom_right'),
        'agent_1': (1, 'team_goal_top_left')
    }

    result = recognizer.recognize_with_team_assignment(
        observations, possible_goals, true_assignment
    )

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Multi-agent goal recognition demo')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes (if training needed)')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['visualize', 'recognize', 'full'],
                        help='Demo mode: visualize only, recognize only, or full')
    args = parser.parse_args()

    print("="*70)
    print("MULTI-AGENT GOAL RECOGNITION SYSTEM DEMO")
    print("="*70)
    print("This demo shows:")
    print("1. Multi-agent environments with team goals")
    print("2. Training individual policies for team goals")
    print("3. Recognizing team assignments and goals from observations")
    print("="*70)

    if args.mode == 'visualize' or args.mode == 'full':
        # Show visualization demos
        demo_scenario_basic()
        input("\nPress Enter to continue to complex scenario...")
        demo_scenario_complex()

    if args.mode == 'recognize' or args.mode == 'full':
        if args.mode == 'full':
            input("\nPress Enter to continue to recognition demo...")
        run_recognition_demo(args.episodes)

    print("\nDemo complete! You can now:")
    print("1. Train agents with more episodes: python train_multi_agent.py --episodes 10000")
    print("2. Run recognition experiments: python recognize_multi_agent.py")
    print("3. Modify team configurations in the environment files")


if __name__ == "__main__":
    main()