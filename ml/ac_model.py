"""Actor-Critic model for PPO."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ACModel(nn.Module):
    """Actor-Critic model for discrete action spaces."""

    def __init__(self, obs_space, action_space, hidden_size=64):
        """Initialize the Actor-Critic model.

        Args:
            obs_space: Observation space of the environment
            action_space: Action space of the environment
            hidden_size: Size of hidden layers
        """
        super().__init__()

        # Get input and output dimensions
        self.obs_dim = obs_space.shape[0]
        self.action_dim = action_space.n

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_dim)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        """Forward pass through the network.

        Args:
            obs: Observation tensor

        Returns:
            dist: Categorical distribution over actions
            value: Value estimate for the state
        """
        # Extract features
        features = self.feature_extractor(obs)

        # Get action logits and value
        action_logits = self.actor(features)
        value = self.critic(features)

        # Create action distribution
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
            features = self.feature_extractor(obs)
            value = self.critic(features)
            return value.cpu().numpy()