#!/usr/bin/env python3
"""
Phase 1: Train RL agents for multi-agent goal recognition.

This script trains separate PPO agents for each possible team goal.
These policies will be used to recognize team membership and goals in multi-agent scenarios.
"""

import os
import argparse

from envs.team_goal_environments import (
    TeamGoalTopRight,
    TeamGoalTopLeft,
    TeamGoalBottomLeft,
    TeamGoalBottomRight
)
from ml.ppo import PPOAgent
from recognizer.multi_agent_recognizer import MultiAgentRecognizer
from metrics.metrics import kl_divergence


def main():
    parser = argparse.ArgumentParser(description='Train agents for multi-agent goal recognition')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes per agent')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda)')
    args = parser.parse_args()

    print("="*70)
    print("MULTI-AGENT TRAINING: PHASE 1")
    print("Training Individual Policies for Team Goals")
    print("="*70)
    print(f"Episodes per agent: {args.episodes}")
    print(f"Models directory: {args.models_dir}")
    print(f"Device: {args.device}")
    print("="*70)

    # Define team goal environments and names
    # These are the possible goals that teams can have
    team_goal_classes = [
        TeamGoalTopRight,    # Team goal: reach top-right
        TeamGoalTopLeft,     # Team goal: reach top-left
        TeamGoalBottomLeft,  # Team goal: reach bottom-left
        TeamGoalBottomRight  # Team goal: reach bottom-right
    ]

    goal_names = [
        "team_goal_top_right",
        "team_goal_top_left",
        "team_goal_bottom_left",
        "team_goal_bottom_right"
    ]

    print(f"\nTeam Goals to Train:")
    for i, goal in enumerate(goal_names, 1):
        print(f"  {i}. {goal}")

    print("\nNote: These are TEAM goals, not individual agent goals.")
    print("During recognition, we'll determine which team each agent belongs to")
    print("based on how well their behavior matches these goal policies.")

    # Create multi-agent recognizer and train agents
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)

    # For training, we don't need team sizes yet - just training individual policies
    recognizer = MultiAgentRecognizer(
        agent_class=PPOAgent,
        goal_env_classes=team_goal_classes,
        models_dir=args.models_dir,
        goal_names=goal_names,
        evaluation_function=kl_divergence,
        num_agents=2,  # Will be configured during recognition
        team_sizes=[1, 1],  # Will be configured during recognition
        episodes=args.episodes,
        train_new=True  # Train new agents
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Models saved to: {args.models_dir}/episodes_{args.episodes}/")
    print("\nTrained policies for team goals:")
    for goal in goal_names:
        model_path = os.path.join(
            args.models_dir,
            f"episodes_{args.episodes}",
            goal,
            "PPOAgent"
        )
        print(f"  - {goal}: {model_path}")

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Run recognition experiments:")
    print("   python recognize.py --episodes %d" % args.episodes)
    print("\n2. Run interactive demo:")
    print("   python demo.py --episodes %d" % args.episodes)
    print("\n3. Run incremental recognition visualizer:")
    print("   python incremental_recognition.py")
    print("\n3. Test different team configurations:")
    print("   - 2 teams with 1 agent each")
    print("   - 2 teams with 2 agents each")
    print("   - Mixed team sizes")
    print("="*70)


if __name__ == "__main__":
    main()
