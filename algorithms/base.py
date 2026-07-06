"""MADRL Agent base class — unified interface for all algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class MADRLAgent(ABC):
    """Base class for all multi-agent RL algorithms."""

    @abstractmethod
    def get_actions(self, obs: np.ndarray, deterministic: bool = False) -> list:
        """Select actions for all agents given observations.

        Args:
            obs: shape (N, obs_dim) or (batch, N, obs_dim)
            deterministic: if True, use greedy action selection

        Returns:
            actions: list of per-agent actions in BDQ format
                - root node: int q
                - non-root: list [q_up0, q_up1, ...]
        """
        pass

    @abstractmethod
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """Store a transition in the agent's buffer.

        Args:
            obs: (N, obs_dim) current observation
            actions: list of per-agent actions (BDQ format)
            rewards: (N,) reward array
            next_obs: (N, obs_dim) next observation
            dones: (N,) done flags
        """
        pass

    @abstractmethod
    def update(self, batch: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """Update algorithm parameters from experience.

        Args:
            batch: optional dict with keys like 'obs', 'actions', 'rewards', etc.

        Returns:
            metrics: dict of training metrics (loss, etc.)
        """
        pass

    @abstractmethod
    def get_entropy(self, obs: np.ndarray) -> np.ndarray:
        """Compute policy entropy for each agent.

        Args:
            obs: (N, obs_dim) observation

        Returns:
            entropy: (N,) entropy per agent
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model parameters to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model parameters from disk."""
        pass
