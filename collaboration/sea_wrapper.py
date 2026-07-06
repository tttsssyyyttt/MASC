"""SEA wrapper — wraps any MADRLAgent with optional expert augmentation."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..algorithms.base import MADRLAgent
from .base import SEABase


class SEAWrapper(MADRLAgent):
    """Wraps a MADRLAgent with optional SEA augmentation.

    When use_sea=False: transparently passes through agent actions.
    When use_sea=True: evaluates intervention and fuses actions.
    """

    def __init__(
        self,
        agent: MADRLAgent,
        sea: Optional[SEABase] = None,
        use_sea: bool = False,
    ):
        self.agent = agent
        self.sea = sea
        self.use_sea = use_sea

    def get_actions(self, obs: np.ndarray, deterministic: bool = False) -> list:
        """Get actions, optionally augmented by SEA expert.

        Args:
            obs: (N, obs_dim)
            deterministic: greedy mode

        Returns:
            actions: list of per-agent actions in BDQ format
        """
        agent_actions = self.agent.get_actions(obs, deterministic)

        if not self.use_sea or self.sea is None:
            return agent_actions

        # Check if expert should intervene
        step = getattr(self.agent, 'total_steps', 0)
        if self.sea.should_intervene(obs, step):
            expert_actions = self.sea.get_expert_action(obs)
            confidence = self._compute_confidence(obs)
            return self.sea.fuse_actions(agent_actions, expert_actions, confidence)

        return agent_actions

    def _compute_confidence(self, obs: np.ndarray) -> float:
        """Compute confidence in the agent's actions.

        Higher entropy → lower confidence. Normalizes to [0, 1].
        Defaults to 0.5 when entropy is unavailable.
        """
        try:
            entropy = self.agent.get_entropy(obs)
            # Entropy per-node can vary; use mean across agents
            mean_entropy = float(np.mean(entropy))
            # Normalize: max_entropy per branch ≈ log(max_order+1)
            # Typical max entropy ~ log(21) ≈ 3.0 for max_order=20
            max_entropy = 3.0
            normalized = np.clip(mean_entropy / max_entropy, 0.0, 1.0)
            return 1.0 - normalized  # high entropy → low confidence
        except Exception:
            return 0.5

    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """Delegate to underlying agent."""
        self.agent.store_transition(obs, actions, rewards, next_obs, dones)

    def get_entropy(self, obs: np.ndarray) -> np.ndarray:
        """Delegate to underlying agent."""
        return self.agent.get_entropy(obs)

    def update(self, batch: Optional[dict] = None) -> dict:
        """Update the underlying agent."""
        return self.agent.update(batch)

    def save(self, path: str):
        """Save the underlying agent."""
        self.agent.save(path)

    def load(self, path: str):
        """Load the underlying agent."""
        self.agent.load(path)
