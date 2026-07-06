"""ALNS solver for inventory/order quantities."""

from __future__ import annotations

from typing import List

import numpy as np

from ..core.config import EnvConfig
from ..core.env import MEIRPEnv
from .ga_inventory_solver import GAInventorySolver


class ALNSInventorySolver(GAInventorySolver):
    """Adaptive large-neighborhood search over per-node order quantities."""

    def __init__(
        self,
        cfg: EnvConfig,
        max_iterations: int = 60,
        destroy_ratio: float = 0.3,
        planning_horizon: int = 5,
        seed: int = 42,
    ):
        super().__init__(
            cfg=cfg,
            population_size=2,
            n_generations=1,
            planning_horizon=planning_horizon,
            seed=seed,
        )
        self.max_iterations = max(1, int(max_iterations))
        self.destroy_ratio = float(destroy_ratio)

    def solve(
        self,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        demand_forecast: np.ndarray,
    ) -> List[List[int]]:
        T = min(self.planning_horizon, int(demand_forecast.shape[0]))
        full_demand = self._expand_demand_forecast(demand_forecast[:T])

        current = self._basestock_solution(inventory, backlog, pipeline, T)
        current_cost = self._evaluate_cost(current, inventory, backlog, pipeline, full_demand)
        best = current.copy()
        best_cost = float(current_cost)
        base_cost = float(current_cost)

        temperature0 = max(1.0, abs(current_cost) * 0.05)
        for it in range(self.max_iterations):
            destroyed = self._destroy(current)
            candidate = self._repair(destroyed, inventory, backlog, pipeline, full_demand)
            candidate_cost = self._evaluate_cost(candidate, inventory, backlog, pipeline, full_demand)

            delta = candidate_cost - current_cost
            temperature = max(1e-6, temperature0 * (1.0 - it / max(self.max_iterations, 1)))
            accept = delta <= 0 or self.rng.random() < np.exp(-delta / temperature)
            if accept:
                current = candidate
                current_cost = float(candidate_cost)
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost

        self.last_best_cost = float(best_cost)
        self.last_basestock_cost = float(base_cost)
        return [best[t].astype(int).tolist() for t in range(T)]

    def get_actions(self, env: MEIRPEnv) -> list:
        forecast = self._default_forecast(env)
        orders = self.solve(env.inventory, env.backlog, env.pipeline, forecast)
        first = orders[0] if orders else [0] * self.N
        return self._orders_to_bdq(first)

    def _destroy(self, solution: np.ndarray) -> np.ndarray:
        destroyed = solution.copy()
        n_entries = max(1, int(round(solution.size * self.destroy_ratio)))
        flat_idx = self.rng.choice(solution.size, size=n_entries, replace=False)
        destroyed.reshape(-1)[flat_idx] = 0
        return destroyed

    def _repair(
        self,
        solution: np.ndarray,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        full_demand: np.ndarray,
    ) -> np.ndarray:
        repaired = solution.copy()
        target = self.cfg.demand_mean * (self.cfg.lead_time + 1)
        for t in range(repaired.shape[0]):
            for i in range(self.N):
                if repaired[t, i] > 0:
                    continue
                if self.rng.random() < 0.5:
                    position = inventory[i] + pipeline[i].sum() - backlog[i]
                    q = max(0, min(int(target - position), self.cfg.max_order))
                else:
                    q = max(0, min(int(full_demand[t, i]), self.cfg.max_order))
                repaired[t, i] = q
        return repaired.astype(int)
