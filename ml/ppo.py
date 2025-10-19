"""PPO (Proximal Policy Optimization) implementation."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, List, Tuple
import gymnasium as gym
from .base_agent import RLAgent
from .ac_model import ACModel


class PPOAgent(RLAgent):
    """PPO agent for discrete action spaces."""

    def __init__(
        self,
        env_name: str,
        models_dir: str,
        goal_hypothesis: str,
        episodes: int = 100000,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 32,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_length: int = 128,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize PPO agent.

        Args:
            env_name: Environment name or class
            models_dir: Directory to save models
            goal_hypothesis: Goal hypothesis name
            episodes: Number of training episodes
            learning_rate: Learning rate
            gamma: Discount factor
            clip_eps: PPO clipping parameter
            epochs: Number of PPO epochs per update
            batch_size: Batch size for updates
            gae_lambda: GAE lambda parameter
            entropy_coef: Entropy coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            rollout_length: Length of rollout before update
            device: Device to use (cpu/cuda)
        """
        super().__init__(
            env_name=env_name,
            models_dir=models_dir,
            goal_hypothesis=goal_hypothesis,
            episodes=episodes,
            learning_rate=learning_rate,
            gamma=gamma,
            **kwargs
        )
        self.learning_rate = learning_rate

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_length = rollout_length

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create environment
        if isinstance(env_name, str):
            self.env = gym.make(env_name)
        else:
            self.env = env_name()

        # Create model
        self.model = ACModel(
            obs_space=self.env.observation_space,
            action_space=self.env.action_space
        ).to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Storage for rollouts
        self.reset_rollout_storage()
        self._current_episode_return = 0.0
        self._current_episode_length = 0
        self.total_steps = 0
        self.total_successes = 0

    def reset_rollout_storage(self):
        """Reset storage for rollout data."""
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []
        self.rollout_episode_returns = []
        self.rollout_episode_lengths = []
        self.rollout_successes = 0

    def collect_rollout(self, num_steps: int) -> Tuple[List, List, List, List, List, List]:
        """Collect rollout data.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Tuple of buffers: (obs, actions, rewards, dones, values, log_probs)
        """
        obs = self.current_obs

        for _ in range(num_steps):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Get action from policy
            with torch.no_grad():
                dist, value = self.model(obs_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # Execute action in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0])
            done = terminated or truncated

            # Store transition
            self.obs_buffer.append(obs)
            self.action_buffer.append(action.cpu().numpy()[0])
            self.reward_buffer.append(reward)
            self.done_buffer.append(done)
            self.value_buffer.append(value.cpu().numpy()[0, 0])
            self.log_prob_buffer.append(log_prob.cpu().numpy()[0])

            # Track state for coverage
            self.track_state(obs)

            # Accumulate episode statistics
            self._current_episode_return += reward
            self._current_episode_length += 1
            self.total_steps += 1

            # Update observation
            obs = next_obs

            # Reset if done
            if done:
                if terminated:
                    self.rollout_successes += 1
                    self.total_successes += 1
                self.rollout_episode_returns.append(self._current_episode_return)
                self.rollout_episode_lengths.append(self._current_episode_length)
                self._current_episode_return = 0.0
                self._current_episode_length = 0
                obs, _ = self.env.reset()
                self.episodes_completed += 1

        self.current_obs = obs

        return (
            self.obs_buffer.copy(),
            self.action_buffer.copy(),
            self.reward_buffer.copy(),
            self.done_buffer.copy(),
            self.value_buffer.copy(),
            self.log_prob_buffer.copy()
        )

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags

        Returns:
            advantages: Advantage estimates
            returns: Return estimates
        """
        advantages = []
        gae = 0

        # Add bootstrap value for last state
        with torch.no_grad():
            last_obs_tensor = torch.FloatTensor(self.current_obs).unsqueeze(0).to(self.device)
            _, last_value = self.model(last_obs_tensor)
            last_value = last_value.cpu().numpy()[0, 0]

        values = values + [last_value]

        # Compute GAE
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.append(gae)

        advantages = advantages[::-1]
        advantages = np.array(advantages)

        # Compute returns
        returns = advantages + np.array(values[:-1])

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_policy(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray
    ):
        """Update policy using PPO.

        Args:
            obs: Observations
            actions: Actions taken
            old_log_probs: Log probabilities of actions under old policy
            advantages: Advantage estimates
            returns: Return estimates
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        action_tensor = torch.LongTensor(actions).to(self.device)
        old_log_prob_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantage_tensor = torch.FloatTensor(advantages).to(self.device)
        return_tensor = torch.FloatTensor(returns).to(self.device)

        # PPO epochs
        for _ in range(self.epochs):
            # Create random batches
            dataset_size = len(obs)
            indices = np.random.permutation(dataset_size)

            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                # Get batch
                batch_obs = obs_tensor[batch_indices]
                batch_actions = action_tensor[batch_indices]
                batch_old_log_probs = old_log_prob_tensor[batch_indices]
                batch_advantages = advantage_tensor[batch_indices]
                batch_returns = return_tensor[batch_indices]

                # Forward pass
                dist, values = self.model(batch_obs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def learn(self):
        """Train the PPO agent."""
        print(f"Training PPO agent for goal: {self.goal_hypothesis}")
        max_steps = getattr(self.env, "max_steps", None)
        runtime_info = f", Max Steps: {max_steps}" if max_steps is not None else ""
        print(
            f"Episodes: {self.episodes}, Device: {self.device}, Rollout: {self.rollout_length}, "
            f"LR: {self.learning_rate}, Gamma: {self.gamma}, Clip: {self.clip_eps}{runtime_info}"
        )

        # Initialize environment
        self.current_obs, _ = self.env.reset()
        self.episodes_completed = 0
        self._current_episode_return = 0.0
        self._current_episode_length = 0
        self.total_steps = 0
        self.total_successes = 0

        # Training loop
        num_updates = self.episodes // (self.rollout_length // 100)  # Approximate

        for update in range(num_updates):
            # Reset rollout storage
            self.reset_rollout_storage()

            # Collect rollout
            obs, actions, rewards, dones, values, log_probs = self.collect_rollout(self.rollout_length)

            # Compute advantages and returns
            advantages, returns = self.compute_gae(rewards, values, dones)

            # Update policy
            self.update_policy(
                np.array(obs),
                np.array(actions),
                np.array(log_probs),
                advantages,
                returns
            )

            # Print progress
            if update % 100 == 0:
                mean_reward = np.mean(rewards)
                if self.rollout_episode_returns:
                    mean_return = np.mean(self.rollout_episode_returns)
                    mean_length = np.mean(self.rollout_episode_lengths)
                    success_rate = self.rollout_successes / len(self.rollout_episode_returns)
                else:
                    mean_return = float(np.sum(rewards))
                    mean_length = float(self._current_episode_length)
                    success_rate = 0.0
                max_steps = getattr(self.env, "max_steps", None)
                max_steps_info = f", Max Steps: {max_steps}" if max_steps is not None else ""
                print(
                    f"Update {update}/{num_updates}, Episodes: {self.episodes_completed}, "
                    f"Mean Reward: {mean_reward:.3f}, "
                    f"Mean Return: {mean_return:.3f}, "
                    f"Success Rate: {success_rate:.2%}, "
                    f"Avg Len: {mean_length:.1f}, "
                    f"Total Successes: {self.total_successes}, "
                    f"Total Steps: {self.total_steps}"
                    f"{max_steps_info}, States Seen: {len(self.states_counter)}"
                )

            # Save model periodically
            if update % 1000 == 0:
                self.save_model()

        # Final save
        self.save_model()
        self.save_states_counter()
        print(f"Training completed. Model saved to {self.model_path}")

    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get action probability distribution for an observation.

        Args:
            observation: Current state observation

        Returns:
            Array of action probabilities
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        probs = self.model.get_action_probabilities(obs_tensor)
        return probs[0]

    def save_model(self):
        """Save the trained model."""
        model_file = os.path.join(self.model_path, "model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episodes_completed': getattr(self, 'episodes_completed', 0),
            'goal_hypothesis': self.goal_hypothesis
        }, model_file)

    def load_model(self):
        """Load a trained model."""
        model_file = os.path.join(self.model_path, "model.pt")
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded model from {model_file}")
        else:
            raise FileNotFoundError(f"No model found at {model_file}")
