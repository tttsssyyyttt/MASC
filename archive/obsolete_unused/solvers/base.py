"""Shared utilities for traditional optimization solvers."""

from __future__ import annotations

from typing import List

import numpy as np

from ..core.config import EnvConfig
from ..core.network import build_network


class BaseSolver:
    """Base class for traditional optimization solvers.

    Provides shared BDQ action formatting and BaseStock heuristic fallback.
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.upstream, self.downstream, self.topo_order, self.terminals = build_network(cfg)

    def format_actions(self, orders_step: List[int]) -> list:
        """Convert solver output (q per node) to BDQ action format.

        Args:
            orders_step: list of N order quantities

        Returns:
            bdq_actions: root -> int, non-root -> list[int]
        """
        actions = []
        for i in range(self.cfg.num_nodes):
            q = orders_step[i]
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

    def basestock_heuristic(
        self,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        n_steps: int = 1,
    ) -> List[List[int]]:
        """BaseStock heuristic fallback for when solver is unavailable.

        Args:
            inventory: (N,) current inventory
            backlog: (N,) current backlog
            pipeline: (N, L) pipeline inventory
            n_steps: number of steps to generate

        Returns:
            orders: list of n_steps lists, each (N,) order quantities
        """
        cfg = self.cfg
        N = cfg.num_nodes
        S = cfg.demand_mean * (cfg.lead_time + 1)

        orders = []
        for _ in range(n_steps):
            step_orders = []
            for i in range(N):
                pipeline_sum = pipeline[i].sum()
                position = inventory[i] + pipeline_sum - backlog[i]
                q = max(0, min(int(S - position), cfg.max_order))
                step_orders.append(q)
            orders.append(step_orders)
        return orders

    def expand_demand_forecast(
        self,
        demand_forecast: np.ndarray,
    ) -> np.ndarray:
        """Map terminal-node demand forecasts to all nodes.

        Non-terminal nodes get an even share of downstream demand.

        Args:
            demand_forecast: (T, n_terminals) demand at terminals

        Returns:
            full_demand: (T, N) demand at all nodes
        """
        T = demand_forecast.shape[0]
        N = self.cfg.num_nodes
        full_demand = np.zeros((T, N))

        for t in range(T):
            for idx, term in enumerate(self.terminals):
                full_demand[t, term] = demand_forecast[t, idx]

        # Propagate upward: non-terminals get average of their downstream
        for i in range(N):
            if i not in self.terminals and self.downstream[i]:
                for t in range(T):
                    ds_demand = [full_demand[t, d] for d in self.downstream[i]]
                    full_demand[t, i] = np.mean(ds_demand) if ds_demand else 0.0

        return full_demand
