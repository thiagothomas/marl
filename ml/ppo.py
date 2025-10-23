"""PPO (Proximal Policy Optimization) implementation."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, List, Tuple, Callable
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
        gamma: float = 0.995,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 256,
        gae_lambda: float = 0.95,
        entropy_coef: float = 3e-3,
        entropy_coef_final: Optional[float] = 1e-3,
        entropy_anneal_fraction: float = 0.8,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_length: int = 256,
        num_envs: int = 8,
        device: Optional[str] = None,
        model_architecture: Optional[str] = None,
        model_hidden_size: Optional[int] = None,
        model_input_dim: Optional[int] = None,
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
            rollout_length: Steps per environment to collect before each update
            num_envs: Number of parallel environments to sample per rollout
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
        self.entropy_coef_initial = float(entropy_coef)
        self.entropy_coef_final = (
            float(entropy_coef) if entropy_coef_final is None else float(entropy_coef_final)
        )
        self.entropy_anneal_fraction = float(np.clip(entropy_anneal_fraction, 0.0, 1.0))
        self.entropy_coef = self.entropy_coef_initial
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_length = rollout_length

        # Store model configuration hints (may be overridden when loading checkpoints)
        self.model_architecture = model_architecture or "expanded"
        self.model_hidden_size = model_hidden_size
        self.model_input_dim = model_input_dim

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_envs = max(1, int(num_envs))

        def _make_env() -> gym.Env:
            if isinstance(env_name, str):
                return gym.make(env_name)
            env_ctor: Callable[[], gym.Env] = env_name
            return env_ctor()

        self.envs = [_make_env() for _ in range(self.num_envs)]
        self.env = self.envs[0]

        # Create model
        self.model = ACModel(
            obs_space=self.env.observation_space,
            action_space=self.env.action_space,
            hidden_size=self.model_hidden_size,
            architecture=self.model_architecture,
            input_dim=self.model_input_dim
        ).to(self.device)
        # Persist the actual parameters the model ended up using.
        self.model_architecture = self.model.architecture
        self.model_hidden_size = self.model.hidden_size
        self.model_input_dim = self.model.input_dim

        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Storage for rollouts
        self.reset_rollout_storage()
        self._current_episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self._current_episode_length = np.zeros(self.num_envs, dtype=np.int32)
        self.total_steps = 0
        self.total_successes = 0
        self.episodes_completed = 0
        self.current_obs = np.stack([env.reset()[0] for env in self.envs])

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

    def collect_rollout(self, num_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collect rollout data.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Tuple of numpy arrays with shapes:
                obs: (num_steps, num_envs, obs_dim)
                actions: (num_steps, num_envs)
                rewards: (num_steps, num_envs)
                dones: (num_steps, num_envs)
                values: (num_steps, num_envs)
                log_probs: (num_steps, num_envs)
        """
        obs_batch = self.current_obs.copy()

        for _ in range(num_steps):
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                dist, value = self.model(obs_tensor)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np = value.squeeze(-1).cpu().numpy()

            rewards_np = np.zeros(self.num_envs, dtype=np.float32)
            dones_np = np.zeros(self.num_envs, dtype=bool)
            next_obs_batch = np.zeros_like(obs_batch)

            for env_idx, env in enumerate(self.envs):
                next_obs, reward, terminated, truncated, _ = env.step(int(actions_np[env_idx]))
                done = bool(terminated or truncated)

                rewards_np[env_idx] = float(reward)
                dones_np[env_idx] = done
                self._current_episode_return[env_idx] += float(reward)
                self._current_episode_length[env_idx] += 1
                self.total_steps += 1

                self.track_state(obs_batch[env_idx])

                if done:
                    if terminated:
                        self.rollout_successes += 1
                        self.total_successes += 1
                    self.rollout_episode_returns.append(self._current_episode_return[env_idx])
                    self.rollout_episode_lengths.append(self._current_episode_length[env_idx])
                    self._current_episode_return[env_idx] = 0.0
                    self._current_episode_length[env_idx] = 0
                    reset_obs, _ = env.reset()
                    next_obs_batch[env_idx] = reset_obs
                    self.episodes_completed += 1
                else:
                    next_obs_batch[env_idx] = next_obs

            self.obs_buffer.append(obs_batch.copy())
            self.action_buffer.append(actions_np.astype(np.int64))
            self.reward_buffer.append(rewards_np.copy())
            self.done_buffer.append(dones_np.copy())
            self.value_buffer.append(values_np.copy())
            self.log_prob_buffer.append(log_probs_np.copy())

            obs_batch = next_obs_batch
            self.current_obs = obs_batch.copy()

        return (
            np.stack(self.obs_buffer),
            np.stack(self.action_buffer),
            np.stack(self.reward_buffer),
            np.stack(self.done_buffer),
            np.stack(self.value_buffer),
            np.stack(self.log_prob_buffer)
        )

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: Reward matrix of shape (num_steps, num_envs)
            values: Value predictions of shape (num_steps, num_envs)
            dones: Done mask of shape (num_steps, num_envs)

        Returns:
            advantages: Flattened, normalized advantages
            returns: Flattened returns aligned with advantages
        """
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        values_arr = np.asarray(values, dtype=np.float32)
        dones_arr = np.asarray(dones, dtype=bool)

        num_steps, num_envs = rewards_arr.shape

        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(self.current_obs, dtype=torch.float32, device=self.device)
            _, last_value_tensor = self.model(last_obs_tensor)
            last_values = last_value_tensor.squeeze(-1).cpu().numpy()

        values_ext = np.concatenate([values_arr, last_values[None, :]], axis=0)

        advantages = np.zeros_like(rewards_arr)
        gae = np.zeros(num_envs, dtype=np.float32)

        for t in reversed(range(num_steps)):
            mask = 1.0 - dones_arr[t].astype(np.float32)
            delta = rewards_arr[t] + self.gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * gae * mask
            advantages[t] = gae

        returns = advantages + values_arr

        flat_advantages = advantages.reshape(-1)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        return flat_advantages, returns.reshape(-1)

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

    def learn(self) -> bool:
        """Train the PPO agent.

        Returns:
            True if training ran to completion, False if interrupted by the user.
        """
        print(f"Training PPO agent for goal: {self.goal_hypothesis}")
        max_steps = getattr(self.env, "max_steps", None)
        runtime_info = f", Max Steps: {max_steps}" if max_steps is not None else ""
        total_batch = self.rollout_length * self.num_envs
        print(
            f"Episodes: {self.episodes}, Device: {self.device}, Envs: {self.num_envs}, "
            f"Rollout/Env: {self.rollout_length}, Batch: {total_batch}, "
            f"LR: {self.learning_rate}, Gamma: {self.gamma}, Clip: {self.clip_eps}{runtime_info}"
        )

        # Initialize environment
        self.current_obs = np.stack([env.reset()[0] for env in self.envs])
        self.episodes_completed = 0
        self._current_episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self._current_episode_length = np.zeros(self.num_envs, dtype=np.int32)
        self.total_steps = 0
        self.total_successes = 0

        rollout_scale = max(1, self.rollout_length // 100)
        num_updates = max(1, self.episodes // rollout_scale)
        if self.entropy_coef_initial != self.entropy_coef_final:
            anneal_updates = max(1, int(num_updates * self.entropy_anneal_fraction))
        else:
            anneal_updates = 0

        interrupted = False

        try:
            for update in range(num_updates):
                if anneal_updates > 0:
                    if update <= anneal_updates:
                        alpha = update / anneal_updates
                        self.entropy_coef = (
                            (1.0 - alpha) * self.entropy_coef_initial
                            + alpha * self.entropy_coef_final
                        )
                    else:
                        self.entropy_coef = self.entropy_coef_final
                # Reset rollout storage
                self.reset_rollout_storage()

                # Collect rollout
                obs, actions, rewards, dones, values, log_probs = self.collect_rollout(self.rollout_length)

                # Compute advantages and returns
                advantages, returns = self.compute_gae(rewards, values, dones)

                # Update policy
                flat_obs = obs.reshape(obs.shape[0] * obs.shape[1], -1)
                flat_actions = actions.reshape(-1)
                flat_log_probs = log_probs.reshape(-1)
                self.update_policy(
                    flat_obs,
                    flat_actions,
                    flat_log_probs,
                    advantages,
                    returns
                )

                # Print progress
                if update % 100 == 0:
                    mean_reward = float(np.mean(rewards))
                    if self.rollout_episode_returns:
                        mean_return = np.mean(self.rollout_episode_returns)
                        mean_length = np.mean(self.rollout_episode_lengths)
                        success_rate = self.rollout_successes / max(1, len(self.rollout_episode_returns))
                    else:
                        mean_return = float(np.sum(rewards))
                        mean_length = float(np.mean(self._current_episode_length))
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
                        f"Total Steps: {self.total_steps}, "
                        f"EntropyCoef: {self.entropy_coef:.4g}"
                        f"{max_steps_info}, States Seen: {len(self.states_counter)}"
                    )

                # Save model periodically
                if update % 200 == 0 and update != 0:
                    self.save_model()
        except KeyboardInterrupt:
            interrupted = True
            print("\nTraining interrupted by user. Leaving existing checkpoint untouched.")
        finally:
            if interrupted:
                # Still persist state visitation stats for analysis, but avoid overwriting weights.
                self.save_states_counter()
                print(f"Training interrupted. Last saved model (if any) remains at {self.model_path}")
            else:
                self.save_model()
                self.save_states_counter()
                print(f"Training completed. Model saved to {self.model_path}")

        return not interrupted

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
            'goal_hypothesis': self.goal_hypothesis,
            'model_architecture': self.model.architecture,
            'model_hidden_size': self.model.hidden_size,
            'model_input_dim': self.model.input_dim,
            'model_obs_dim': self.model.obs_dim,
            'entropy_coef_initial': self.entropy_coef_initial,
            'entropy_coef_final': self.entropy_coef_final,
            'entropy_anneal_fraction': self.entropy_anneal_fraction,
        }, model_file)

    def load_model(self):
        """Load a trained model."""
        model_file = os.path.join(self.model_path, "model.pt")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"No model found at {model_file}")

        checkpoint = torch.load(model_file, map_location=self.device)
        state_dict = checkpoint['model_state_dict']

        architecture = checkpoint.get('model_architecture')
        hidden_size = checkpoint.get('model_hidden_size')
        input_dim = checkpoint.get('model_input_dim')

        if architecture is None or hidden_size is None or input_dim is None:
            architecture, hidden_size, input_dim = ACModel.infer_config_from_state_dict(state_dict)

        # Rebuild the model with the appropriate configuration.
        self.model = ACModel(
            obs_space=self.env.observation_space,
            action_space=self.env.action_space,
            hidden_size=hidden_size,
            architecture=architecture,
            input_dim=input_dim
        ).to(self.device)

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as err:
            inferred_architecture, inferred_hidden_size, inferred_input_dim = ACModel.infer_config_from_state_dict(state_dict)
            if (
                inferred_architecture != architecture
                or inferred_hidden_size != hidden_size
                or inferred_input_dim != input_dim
            ):
                self.model = ACModel(
                    obs_space=self.env.observation_space,
                    action_space=self.env.action_space,
                    hidden_size=inferred_hidden_size,
                    architecture=inferred_architecture,
                    input_dim=inferred_input_dim
                ).to(self.device)
                self.model.load_state_dict(state_dict)
                architecture = inferred_architecture
                hidden_size = inferred_hidden_size
                input_dim = inferred_input_dim
            else:
                raise err

        self.entropy_coef_initial = float(
            checkpoint.get('entropy_coef_initial', self.entropy_coef_initial)
        )
        self.entropy_coef_final = float(
            checkpoint.get('entropy_coef_final', self.entropy_coef_final)
        )
        self.entropy_anneal_fraction = float(
            checkpoint.get('entropy_anneal_fraction', self.entropy_anneal_fraction)
        )
        self.entropy_coef = self.entropy_coef_final

        # Persist resolved configuration.
        self.model_architecture = architecture
        self.model_hidden_size = hidden_size
        self.model_input_dim = input_dim

        # Recreate optimizer so its parameter groups match the model.
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        optimizer_state = checkpoint.get('optimizer_state_dict')
        if optimizer_state:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except (ValueError, RuntimeError):
                print(
                    "Warning: optimizer state mismatch detected; using freshly initialized optimizer."
                )

        self.episodes_completed = checkpoint.get(
            'episodes_completed',
            getattr(self, 'episodes_completed', 0)
        )

        print(
            f"Loaded model from {model_file} "
            f"(architecture={architecture}, hidden_size={hidden_size}, input_dim={input_dim})"
        )
