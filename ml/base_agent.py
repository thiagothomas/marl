"""Base RL agent class for goal recognition."""

from abc import ABC, abstractmethod
import os
import pickle
from typing import Any, Dict, Optional, Tuple
import numpy as np


class RLAgent(ABC):
    """Abstract base class for all RL agents."""

    def __init__(
        self,
        env_name: str,
        models_dir: str,
        goal_hypothesis: str,
        episodes: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        **kwargs
    ):
        """Initialize the RL agent.

        Args:
            env_name: Name of the environment
            models_dir: Directory to save/load models
            goal_hypothesis: The goal this agent is trained for
            episodes: Number of training episodes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            **kwargs: Additional algorithm-specific parameters
        """
        self.env_name = env_name
        self.models_dir = models_dir
        self.goal_hypothesis = goal_hypothesis
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Create model directory
        self.model_path = os.path.join(
            models_dir,
            f"episodes_{episodes}",
            goal_hypothesis,
            self.__class__.__name__
        )
        os.makedirs(self.model_path, exist_ok=True)

        # Track unique states visited during training
        self.states_counter = {}

    @abstractmethod
    def learn(self):
        """Train the agent on the environment."""
        pass

    @abstractmethod
    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get action probability distribution for a given observation.

        Args:
            observation: Current state observation

        Returns:
            Array of action probabilities
        """
        pass

    @abstractmethod
    def save_model(self):
        """Save the trained model to disk."""
        pass

    @abstractmethod
    def load_model(self):
        """Load a trained model from disk."""
        pass

    def save_states_counter(self):
        """Save the states visited during training."""
        states_path = os.path.join(self.model_path, "states_seen.pkl")
        with open(states_path, 'wb') as f:
            pickle.dump(self.states_counter, f)

    def load_states_counter(self):
        """Load the states visited during training."""
        states_path = os.path.join(self.model_path, "states_seen.pkl")
        if os.path.exists(states_path):
            with open(states_path, 'rb') as f:
                self.states_counter = pickle.load(f)

    def track_state(self, state: np.ndarray):
        """Track a visited state."""
        state_str = str(state.tolist())
        self.states_counter[state_str] = self.states_counter.get(state_str, 0) + 1