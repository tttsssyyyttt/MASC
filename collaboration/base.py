"""SEA (Selective Expert Augmentation) interface definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class SEABase(ABC):
    """Base interface for Selective Expert Augmentation.

    Three core methods:
    - should_intervene: decide if expert should act this step
    - get_expert_action: compute expert's recommended action
    - fuse_actions: blend agent and expert actions
    """

    @abstractmethod
    def should_intervene(self, obs: np.ndarray, step: int) -> bool:
        """Decide whether the expert should intervene.

        Args:
            obs: (N, obs_dim) current observations
            step: current timestep

        Returns:
            True if expert should intervene
        """
        pass

    @abstractmethod
    def get_expert_action(self, obs: np.ndarray) -> list:
        """Get expert's recommended action.

        Args:
            obs: (N, obs_dim) current observations

        Returns:
            actions: list of (q, alpha) per agent
        """
        pass

    @abstractmethod
    def fuse_actions(
        self,
        agent_actions: list,
        expert_actions: list,
        confidence: float,
    ) -> list:
        """Blend agent and expert actions.

        Args:
            agent_actions: list of (q, alpha) from RL agent
            expert_actions: list of (q, alpha) from expert
            confidence: confidence score in [0, 1]

        Returns:
            fused_actions: list of (q, alpha)
        """
        pass
