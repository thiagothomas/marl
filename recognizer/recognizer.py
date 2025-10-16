"""Goal recognizer implementation."""

import os
import numpy as np
from typing import List, Tuple, Optional, Callable, Type, Dict, Any
from ml.base_agent import RLAgent
from ml.ppo import PPOAgent
from metrics.metrics import softmin
import gymnasium as gym


class Recognizer:
    """Goal recognizer using trained RL agents."""

    def __init__(
        self,
        agent_class: Type[RLAgent],
        env_classes: List,
        models_dir: str,
        goal_names: List[str],
        evaluation_function: Callable,
        episodes: int = 10000,
        train_new: bool = True
    ):
        """Initialize the goal recognizer.

        Args:
            agent_class: Class of RL agent to use (e.g., PPOAgent)
            env_classes: List of environment classes for each goal
            models_dir: Directory to save/load models
            goal_names: Names of goals corresponding to environments
            evaluation_function: Function to evaluate observation match
            episodes: Number of training episodes
            train_new: Whether to train new agents or load existing ones
        """
        self.agent_class = agent_class
        self.env_classes = env_classes
        self.models_dir = models_dir
        self.goal_names = goal_names
        self.evaluation_function = evaluation_function
        self.episodes = episodes

        # Create or load agents for each goal
        self.agents = []
        for env_class, goal_name in zip(env_classes, goal_names):
            agent = agent_class(
                env_name=env_class,
                models_dir=models_dir,
                goal_hypothesis=goal_name,
                episodes=episodes
            )

            if train_new:
                # Train the agent (Phase 1)
                print(f"\n{'='*60}")
                print(f"Training agent for goal: {goal_name}")
                print(f"{'='*60}")
                agent.learn()
            else:
                # Load pre-trained agent (Phase 2)
                print(f"Loading pre-trained agent for goal: {goal_name}")
                agent.load_model()
                agent.load_states_counter()

            self.agents.append(agent)

    def collect_observations(
        self,
        env: gym.Env,
        num_steps: int = 50,
        policy: Optional[Callable] = None
    ) -> List[Tuple[np.ndarray, int]]:
        """Collect observations from an agent acting in the environment.

        Args:
            env: Environment to collect observations from
            num_steps: Number of steps to collect
            policy: Optional policy to use (if None, uses random policy)

        Returns:
            List of (observation, action) tuples
        """
        observations = []
        obs, _ = env.reset()

        for _ in range(num_steps):
            if policy is not None:
                # Use provided policy
                action = policy(obs)
            else:
                # Random policy
                action = env.action_space.sample()

            observations.append((obs.copy(), action))

            next_obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

        return observations

    def recognize_goal(
        self,
        observations: List[Tuple[np.ndarray, int]],
        real_goal_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Recognize the goal from observations.

        This is the core recognition algorithm (Phase 2).

        Args:
            observations: List of (state, action) tuples from observed agent
            real_goal_idx: Index of the real goal (for evaluation)

        Returns:
            Dictionary with recognition results
        """
        scores = []

        # Evaluate each trained agent against the observations
        print("\n" + "="*60)
        print("GOAL RECOGNITION PHASE")
        print("="*60)

        for i, agent in enumerate(self.agents):
            # Calculate how well this agent's policy explains the observations
            score = self.evaluation_function(observations, agent)
            scores.append(score)
            print(f"Goal '{self.goal_names[i]}' score: {score:.4f}")

        # Convert scores to probabilities (lower score = higher probability)
        scores = np.array(scores)
        probabilities = softmin(scores)

        # Rank goals by probability
        rankings = sorted(
            enumerate(probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        # Get predicted goal
        predicted_idx = rankings[0][0]
        predicted_goal = self.goal_names[predicted_idx]
        confidence = rankings[0][1]

        print("\n" + "-"*60)
        print("RECOGNITION RESULTS:")
        print("-"*60)
        for rank, (idx, prob) in enumerate(rankings, 1):
            marker = "<<<" if idx == predicted_idx else ""
            print(f"{rank}. {self.goal_names[idx]}: {prob:.3f} {marker}")

        # Check if prediction is correct (if real goal provided)
        is_correct = None
        if real_goal_idx is not None:
            is_correct = predicted_idx == real_goal_idx
            real_goal = self.goal_names[real_goal_idx]
            print(f"\nReal goal: {real_goal}")
            print(f"Predicted: {predicted_goal}")
            print(f"Correct: {'YES' if is_correct else 'NO'}")

        return {
            'predicted_goal': predicted_goal,
            'predicted_idx': predicted_idx,
            'confidence': confidence,
            'probabilities': probabilities,
            'rankings': rankings,
            'scores': scores,
            'is_correct': is_correct
        }

    def evaluate_on_test_set(
        self,
        test_environments: List[gym.Env],
        test_goal_indices: List[int],
        num_steps_per_env: int = 50
    ) -> Dict[str, Any]:
        """Evaluate recognition accuracy on multiple test environments.

        Args:
            test_environments: List of test environments
            test_goal_indices: Ground truth goal indices
            num_steps_per_env: Number of observation steps per environment

        Returns:
            Evaluation results
        """
        correct_predictions = 0
        all_results = []

        print("\n" + "="*60)
        print("EVALUATION ON TEST SET")
        print("="*60)

        for i, (env, true_goal_idx) in enumerate(zip(test_environments, test_goal_indices)):
            print(f"\nTest case {i+1}/{len(test_environments)}")
            print(f"True goal: {self.goal_names[true_goal_idx]}")

            # Collect observations from test environment
            observations = self.collect_observations(env, num_steps_per_env)

            # Recognize goal
            result = self.recognize_goal(observations, true_goal_idx)
            all_results.append(result)

            if result['is_correct']:
                correct_predictions += 1

        # Calculate accuracy
        accuracy = correct_predictions / len(test_environments)

        print("\n" + "="*60)
        print(f"FINAL ACCURACY: {accuracy:.2%} ({correct_predictions}/{len(test_environments)})")
        print("="*60)

        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_tests': len(test_environments),
            'all_results': all_results
        }