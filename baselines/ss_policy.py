"""(s, S) inventory policy.

Rule: if position < s, order up to S; otherwise order 0.
    position = inventory + pipeline_sum - backlog
    if position < s: q = S - position
    else: q = 0

When multiple upstreams exist, split evenly.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.config import EnvConfig
from core.env import MEIRPEnv
from .base import InventoryPolicy


class SSPolicy(InventoryPolicy):
    """(s, S) inventory policy for all nodes."""

    def __init__(
        self,
        cfg: EnvConfig,
        reorder_points: Optional[List[float]] = None,
        order_up_to_levels: Optional[List[float]] = None,
    ):
        super().__init__(cfg)

        # Default: s = demand_mean * lead_time, S = demand_mean * (lead_time + 1)
        if reorder_points is not None:
            self.s = np.array(reorder_points, dtype=np.float64)
        else:
            self.s = np.full(cfg.num_nodes, cfg.demand_mean * cfg.lead_time)

        if order_up_to_levels is not None:
            self.S = np.array(order_up_to_levels, dtype=np.float64)
        else:
            self.S = np.full(cfg.num_nodes, cfg.demand_mean * (cfg.lead_time + 1))

    def get_actions(self, env: MEIRPEnv):
        """Compute actions given current env state.

        Returns:
            actions: list of per-agent actions (BDQ format)
                - root nodes: int q
                - non-root nodes: list [q_up0, q_up1, ...]
        """
        actions = []
        for i in range(self.cfg.num_nodes):
            pipeline_sum = env.pipeline[i].sum()
            position = env.inventory[i] + pipeline_sum - env.backlog[i]

            if position < self.s[i]:
                q = max(0, int(self.S[i] - position))
            else:
                q = 0

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

        return actions
