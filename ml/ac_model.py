"""Actor-Critic model for PPO."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ACModel(nn.Module):
    """Actor-Critic model for discrete action spaces.

    Supports multiple architecture layouts to keep old checkpoints loadable while
    allowing richer networks for new training runs.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        hidden_size: Optional[int] = None,
        architecture: str = "expanded",
        input_dim: Optional[int] = None
    ) -> None:
        """Initialize the Actor-Critic model.

        Args:
            obs_space: Observation space of the environment
            action_space: Action space of the environment
            hidden_size: Size of hidden layers (architecture dependent)
            architecture: Network layout identifier ('expanded' or 'legacy')
            input_dim: Override for the expected flattened observation dimension
        """
        super().__init__()

        # Keep track of both actual and expected observation dimensions.
        self.obs_dim = obs_space.shape[0]
        self.input_dim = input_dim if input_dim is not None else self.obs_dim
        self.action_dim = action_space.n
        self.architecture = architecture

        if architecture == "expanded":
            self.hidden_size = hidden_size or 128
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU()
            )
            self.actor = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.action_dim)
            )
            self.critic = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1)
            )
        elif architecture == "legacy":
            self.hidden_size = hidden_size or 64
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_size),
                nn.ReLU()
            )
            self.actor = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.action_dim)
            )
            self.critic = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1)
            )
        else:
            raise ValueError(f"Unsupported ACModel architecture '{architecture}'.")

    def _prepare_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Adjust observation tensor to the expected input dimension."""
        if obs.shape[-1] == self.input_dim:
            return obs

        if obs.shape[-1] > self.input_dim:
            return obs[..., :self.input_dim]

        pad_size = self.input_dim - obs.shape[-1]
        padding_shape = (*obs.shape[:-1], pad_size)
        padding = obs.new_zeros(padding_shape)
        return torch.cat([obs, padding], dim=-1)

    def forward(self, obs):
        """Forward pass through the network.

        Args:
            obs: Observation tensor

        Returns:
            dist: Categorical distribution over actions
            value: Value estimate for the state
        """
        obs = self._prepare_obs(obs)
        features = self.feature_extractor(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        dist = Categorical(logits=action_logits)
        return dist, value

    def get_action_probabilities(self, obs):
        """Get action probabilities for an observation.

        Args:
            obs: Observation tensor

        Returns:
            Action probabilities as numpy array
        """
        with torch.no_grad():
            obs = self._prepare_obs(obs)
            features = self.feature_extractor(obs)
            action_logits = self.actor(features)
            probs = F.softmax(action_logits, dim=-1)
            return probs.cpu().numpy()

    def get_value(self, obs):
        """Get value estimate for an observation.

        Args:
            obs: Observation tensor

        Returns:
            Value estimate
        """
        with torch.no_grad():
            obs = self._prepare_obs(obs)
            features = self.feature_extractor(obs)
            value = self.critic(features)
            return value.cpu().numpy()

    @staticmethod
    def infer_config_from_state_dict(
        state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[str, int, int]:
        """Infer architecture metadata from a checkpoint state dict."""
        first_weight = state_dict.get("feature_extractor.0.weight")
        if first_weight is None:
            raise ValueError("Unable to infer architecture: missing feature_extractor.0.weight")

        hidden_size = first_weight.shape[0]
        input_dim = first_weight.shape[1]
        if "feature_extractor.2.weight" in state_dict:
            return "expanded", hidden_size, input_dim
        return "legacy", hidden_size, input_dim
