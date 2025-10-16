"""Evaluation metrics for goal recognition."""

import numpy as np
from typing import List, Tuple
from ml.base_agent import RLAgent


def softmin(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply softmin to convert distances/costs to probabilities.

    Lower scores get higher probabilities.

    Args:
        scores: Array of scores (lower is better)
        temperature: Temperature parameter for softmin

    Returns:
        Probability distribution
    """
    # Negate scores for softmax (since we want softmin)
    neg_scores = -scores / temperature
    # Subtract max for numerical stability
    neg_scores = neg_scores - np.max(neg_scores)
    exp_scores = np.exp(neg_scores)
    return exp_scores / np.sum(exp_scores)


def kl_divergence(
    observations: List[Tuple[np.ndarray, int]],
    agent: RLAgent
) -> float:
    """Calculate KL divergence between observed and predicted actions.

    Args:
        observations: List of (state, action) tuples
        agent: Trained agent to evaluate

    Returns:
        Mean KL divergence
    """
    kl_divs = []

    for obs, observed_action in observations:
        # Get predicted action probabilities
        predicted_probs = agent.get_action_probabilities(obs)

        # Create one-hot encoding for observed action
        observed_probs = np.zeros_like(predicted_probs)
        observed_probs[observed_action] = 1.0

        # Calculate KL divergence
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        predicted_probs = np.clip(predicted_probs, epsilon, 1.0)

        kl_div = np.sum(observed_probs * np.log(observed_probs / predicted_probs + epsilon))
        kl_divs.append(kl_div)

    return np.mean(kl_divs)


def mean_action_distance(
    observations: List[Tuple[np.ndarray, int]],
    agent: RLAgent
) -> float:
    """Calculate mean distance between observed and predicted actions.

    Args:
        observations: List of (state, action) tuples
        agent: Trained agent to evaluate

    Returns:
        Mean action distance
    """
    distances = []

    for obs, observed_action in observations:
        # Get predicted action probabilities
        predicted_probs = agent.get_action_probabilities(obs)

        # Get most likely predicted action
        predicted_action = np.argmax(predicted_probs)

        # Calculate distance (0 if same, 1 if different)
        distance = float(predicted_action != observed_action)
        distances.append(distance)

    return np.mean(distances)


def cross_entropy(
    observations: List[Tuple[np.ndarray, int]],
    agent: RLAgent
) -> float:
    """Calculate cross-entropy between observed and predicted actions.

    Args:
        observations: List of (state, action) tuples
        agent: Trained agent to evaluate

    Returns:
        Mean cross-entropy
    """
    ce_values = []

    for obs, observed_action in observations:
        # Get predicted action probabilities
        predicted_probs = agent.get_action_probabilities(obs)

        # Calculate cross-entropy for the observed action
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        prob = np.clip(predicted_probs[observed_action], epsilon, 1.0)
        ce = -np.log(prob)
        ce_values.append(ce)

    return np.mean(ce_values)


def trajectory_likelihood(
    observations: List[Tuple[np.ndarray, int]],
    agent: RLAgent
) -> float:
    """Calculate the likelihood of the trajectory under the agent's policy.

    Args:
        observations: List of (state, action) tuples
        agent: Trained agent to evaluate

    Returns:
        Negative log likelihood of trajectory
    """
    log_likelihood = 0.0

    for obs, observed_action in observations:
        # Get predicted action probabilities
        predicted_probs = agent.get_action_probabilities(obs)

        # Add log probability of observed action
        epsilon = 1e-10
        prob = np.clip(predicted_probs[observed_action], epsilon, 1.0)
        log_likelihood += np.log(prob)

    # Return negative log likelihood (lower is better for our metrics)
    return -log_likelihood