"""Expert Advisor — BaseStock-based expert for SEA framework."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import SEABase
from .budget import BudgetManager
from ..core.config import EnvConfig
from ..core.network import build_network


class ExpertAdvisor(SEABase):
    """Expert advisor using BaseStock heuristic as the expert.

    Can be replaced with MILP-based advisor by injecting a different
    get_expert_action implementation.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        intervention_threshold: float = 0.3,
        budget: Optional[int] = None,
        cooldown: int = 5,
    ):
        self.cfg = cfg
        self.intervention_threshold = intervention_threshold
        self.upstream, _, _, _ = build_network(cfg)

        # Default base stock level = demand_mean * (lead_time + 1) per node
        self.S = np.full(cfg.num_nodes, cfg.demand_mean * (cfg.lead_time + 1))

        # Budget manager for intervention control
        self.budget_mgr = BudgetManager(
            total_budget=budget if budget is not None else cfg.episode_len,
            cooldown=cooldown,
        )

        # Precompute denormalization factors (matching _get_obs in env.py)
        self._max_inv = max(cfg.init_inventory * 3.0, cfg.max_order * cfg.lead_time * 2)
        self._max_dem = cfg.demand_mean * 4.0
        self._max_pipe = self._max_dem * cfg.lead_time
        self.L = cfg.lead_time

    def should_intervene(self, obs: np.ndarray, step: int) -> bool:
        """Check budget + cooldown, then use backlog-based heuristic."""
        if not self.budget_mgr.can_intervene(step):
            return False

        # Denormalize to get backlog ratio
        backlog = obs[:, 1] * self._max_inv
        inventory = obs[:, 0] * self._max_inv
        # Intervene when mean backlog exceeds threshold fraction of mean inventory
        mean_inv = np.mean(np.maximum(inventory, 0))
        mean_bl = np.mean(np.maximum(backlog, 0))
        if mean_inv < 1e-6:
            return mean_bl > 1.0
        ratio = mean_bl / mean_inv
        return ratio > self.intervention_threshold

    def get_expert_action(self, obs: np.ndarray) -> list:
        """Compute BaseStock actions from normalized observations.

        Denormalizes obs to reconstruct inventory/backlog/pipeline state,
        then applies the order-up-to rule.
        """
        # Denormalize
        inventory = obs[:, 0] * self._max_inv
        backlog = obs[:, 1] * self._max_inv
        pipeline = obs[:, 3:3 + self.L] * self._max_pipe

        actions = []
        for i in range(self.cfg.num_nodes):
            pipeline_sum = pipeline[i].sum()
            position = inventory[i] + pipeline_sum - backlog[i]
            q = max(0, int(self.S[i] - position))
            q = min(q, self.cfg.max_order)

            k = len(self.upstream[i])
            if k == 0:
                actions.append(q)
            else:
                q_per_up = q // k
                remainder = q % k
                action_i = [q_per_up] * k
                for j in range(remainder):
                    action_i[j] += 1
                actions.append(action_i)

        self.budget_mgr.record_intervention(0)
        return actions

    def fuse_actions(
        self,
        agent_actions: list,
        expert_actions: list,
        confidence: float,
    ) -> list:
        """Blend agent and expert actions based on confidence.

        confidence > 0.5 → use expert; otherwise use agent.
        This provides a clean handoff between learned and heuristic behavior.
        """
        if confidence > 0.5:
            return list(expert_actions)
        return list(agent_actions)
