"""MILP-based expert advisor — stand-in for "human expert" in SEA.

MILP is treated as a stronger expert than BaseStock because:
  - It solves a rolling-horizon optimization problem
  - With true demand forecast, it serves as an oracle upper bound
  - It achieves higher fill rates than BaseStock alone

In the SEA framework:
  - BaseStock = algorithm expert (fast, handles routine issues)
  - MILP = "human expert" stand-in (stronger, handles extreme cases)

When real human experiments are conducted, the MILP can be replaced
by actual human decision-makers through the same interface.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.config import EnvConfig
from solvers.milp_solver import MILPSolver


class MILPAdvisor:
    """MILP expert advisor that can act as a 'human expert' proxy in SEA.

    In the escalation hierarchy, MILP is invoked only for extreme cases
    (when both SPC and BaseStock detect something unusual).
    """

    def __init__(self, cfg: EnvConfig, planning_horizon: int = 5):
        self.cfg = cfg
        self.milp = MILPSolver(cfg, planning_horizon=planning_horizon)
        self.planning_horizon = planning_horizon

        from ..core.network import build_network
        self.upstream, _, _, self.terminals = build_network(cfg)

    def get_action(self, inventory, backlog, pipeline, demand_forecast=None) -> list:
        """Get MILP expert action for current state.

        Args:
            inventory: (N,) current inventory
            backlog: (N,) current backlog
            pipeline: (N, L) pipeline
            demand_forecast: (T, n_terminals) forecast, auto-generated if None

        Returns:
            bdq_actions: list in BDQ format (root->int, non-root->list[int])
        """
        n_terms = len(self.terminals)

        if demand_forecast is None:
            demand_forecast = np.random.poisson(
                self.cfg.demand_mean,
                size=(self.planning_horizon, n_terms)
            ).astype(float)

        # V2: get_actions() now returns BDQ format directly
        bdq_actions = self.milp.get_actions(
            inventory, backlog, pipeline, demand_forecast
        )

        # Ensure proper types
        typed_actions = []
        for i, a in enumerate(bdq_actions):
            k = len(self.upstream[i])
            if k == 0:
                typed_actions.append(min(int(a), self.cfg.max_order))
            else:
                typed_actions.append([min(int(x), self.cfg.max_order) for x in a])

        return typed_actions
