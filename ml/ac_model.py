"""Actor-Critic model for PPO."""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gymnasium import spaces


class MultiCategorical:
    """Collection of independent categorical dists acting like a single policy head."""

    def __init__(self, logits: Sequence[torch.Tensor]) -> None:
        self._dists = [Categorical(logits=logit) for logit in logits]

    def sample(self) -> torch.Tensor:
        samples = [dist.sample() for dist in self._dists]
        return torch.stack(samples, dim=-1)

    def mode(self) -> torch.Tensor:
        modes = [torch.argmax(dist.logits, dim=-1) for dist in self._dists]
        return torch.stack(modes, dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        parts = []
        for idx, dist in enumerate(self._dists):
            parts.append(dist.log_prob(actions[..., idx]))
        return torch.stack(parts, dim=-1).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        parts = [dist.entropy() for dist in self._dists]
        return torch.stack(parts, dim=-1).sum(dim=-1)

    def probs(self) -> torch.Tensor:
        probs = [F.softmax(dist.logits, dim=-1) for dist in self._dists]
        return torch.stack(probs, dim=1)


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
        self.is_multi_discrete = isinstance(action_space, spaces.MultiDiscrete)
        if self.is_multi_discrete:
            self.action_dims: List[int] = [int(n) for n in action_space.nvec.tolist()]
            self.action_dim = sum(self.action_dims)
        else:
            self.action_dims = []
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
        if self.is_multi_discrete:
            splits = torch.split(action_logits, self.action_dims, dim=-1)
            dist = MultiCategorical(splits)
        else:
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
            if self.is_multi_discrete:
                splits = torch.split(action_logits, self.action_dims, dim=-1)
                probs = torch.stack([
                    F.softmax(split, dim=-1) for split in splits
                ], dim=1)
            else:
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
