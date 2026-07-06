"""BaseStock inventory policy.

Rule: order up to S
    q = max(0, S - inventory - pipeline_sum - backlog)

When multiple upstreams exist, split evenly.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.config import EnvConfig
from core.env import MEIRPEnv
from .base import InventoryPolicy


class BaseStockPolicy(InventoryPolicy):
    """BaseStock (order-up-to) policy for all nodes."""

    def __init__(self, cfg: EnvConfig, base_stock_levels: Optional[List[float]] = None):
        super().__init__(cfg)

        # Default base stock level = demand_mean * (lead_time + 1) per node
        if base_stock_levels is not None:
            self.S = np.array(base_stock_levels, dtype=np.float64)
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

        return actions
