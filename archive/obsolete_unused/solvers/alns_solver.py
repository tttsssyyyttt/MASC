"""ALNS (Adaptive Large Neighborhood Search) Solver for MEIRP.

Iteratively destroys and repairs solutions using adaptive operator selection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.config import EnvConfig
from ..core.network import build_network


class ALNSSolver:
    """ALNS solver for MEIRP.

    Uses destroy operators (remove some orders) and repair operators
    (re-fill orders) with adaptive weight adjustment.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        max_iterations: int = 200,
        destroy_ratio: float = 0.3,
        n_destroy_operators: int = 3,
        n_repair_operators: int = 3,
        planning_horizon: int = 5,
        seed: int = 42,
    ):
        self.cfg = cfg
        self.max_iterations = max_iterations
        self.destroy_ratio = destroy_ratio
        self.n_destroy_ops = n_destroy_operators
        self.n_repair_ops = n_repair_operators
        self.planning_horizon = planning_horizon
        self.rng = np.random.default_rng(seed)

        self.upstream, self.downstream, self.topo_order, self.terminals = build_network(cfg)
        self.N = cfg.num_nodes
        self.L = cfg.lead_time

        # Operator weights (adaptive)
        self.destroy_weights = np.ones(n_destroy_operators)
        self.repair_weights = np.ones(n_repair_operators)

    def solve(
        self,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        demand_forecast: np.ndarray,
    ) -> List[List[int]]:
        """Solve using ALNS.

        Args:
            inventory: (N,) current inventory
            backlog: (N,) current backlog
            pipeline: (N, L) in-transit orders
            demand_forecast: (T, n_terminals) demand forecast

        Returns:
            orders: (T, N) order quantities per timestep per node
        """
        T = min(self.planning_horizon, demand_forecast.shape[0])
        full_demand = self._expand_demand_forecast(demand_forecast[:T])

        # Initial solution: BaseStock heuristic
        current = self._basestock_solution(inventory, backlog, pipeline, T)
        current_cost = self._evaluate_cost(current, inventory, backlog, pipeline, full_demand)

        best = current.copy()
        best_cost = current_cost

        # Segment-based weight adaptation
        segment_size = max(1, self.max_iterations // 10)
        destroy_scores = np.zeros(self.n_destroy_ops)
        repair_scores = np.zeros(self.n_repair_ops)
        destroy_counts = np.zeros(self.n_destroy_ops)
        repair_counts = np.zeros(self.n_repair_ops)

        for it in range(self.max_iterations):
            # Select operators
            d_idx = self._select_operator(self.destroy_weights)
            r_idx = self._select_operator(self.repair_weights)

            destroy_counts[d_idx] += 1
            repair_counts[r_idx] += 1

            # Apply destroy
            destroyed = self._destroy(current, d_idx, T)

            # Apply repair
            candidate = self._repair(destroyed, r_idx, inventory, backlog, pipeline, full_demand, T)
            candidate_cost = self._evaluate_cost(candidate, inventory, backlog, pipeline, full_demand)

            # Acceptance criterion (simulated annealing-like)
            delta = candidate_cost - current_cost
            temp = max(1.0, best_cost * 0.01) * (1 - it / self.max_iterations)
            accept = delta < 0 or np.exp(-delta / temp) > self.rng.random()

            if accept:
                current = candidate
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
                    destroy_scores[d_idx] += 3  # best improvement
                    repair_scores[r_idx] += 3
                else:
                    destroy_scores[d_idx] += 1  # new solution
                    repair_scores[r_idx] += 1
            else:
                destroy_scores[d_idx] += 0
                repair_scores[r_idx] += 0

            # Periodic weight update
            if (it + 1) % segment_size == 0:
                self._update_weights(destroy_scores, destroy_counts, self.destroy_weights)
                self._update_weights(repair_scores, repair_counts, self.repair_weights)
                destroy_scores[:] = 0
                repair_scores[:] = 0
                destroy_counts[:] = 0
                repair_counts[:] = 0

        orders = []
        for t in range(T):
            orders.append(best[t].tolist())

        return orders

    def _select_operator(self, weights: np.ndarray) -> int:
        """Roulette wheel selection."""
        probs = weights / weights.sum()
        return self.rng.choice(len(weights), p=probs)

    def _update_weights(self, scores: np.ndarray, counts: np.ndarray, weights: np.ndarray):
        """Update operator weights based on scores."""
        for i in range(len(weights)):
            if counts[i] > 0:
                weights[i] = 0.7 * weights[i] + 0.3 * (scores[i] / counts[i])

    def _destroy(self, solution: np.ndarray, op_idx: int, T: int) -> np.ndarray:
        """Destroy part of the solution."""
        destroyed = solution.copy()
        n_destroy = max(1, int(self.N * T * self.destroy_ratio))

        if op_idx == 0:
            # Random destroy
            for _ in range(n_destroy):
                t = self.rng.integers(0, T)
                i = self.rng.integers(0, self.N)
                destroyed[t, i] = 0
        elif op_idx == 1:
            # Worst destroy: remove orders at nodes with highest inventory
            for t in range(T):
                costs = np.zeros(self.N)
                for i in range(self.N):
                    costs[i] = destroyed[t, i]  # proxy: higher order = more cost
                worst = costs.argsort()[::-1][:max(1, int(self.N * self.destroy_ratio))]
                destroyed[t, worst] = 0
        else:
            # Shaw destroy: remove related (same-layer) nodes
            for t in range(T):
                layer = self.rng.integers(0, max(self.cfg.layers) + 1)
                layer_nodes = [i for i in range(self.N) if self.cfg.layers[i] == layer]
                for i in layer_nodes:
                    if self.rng.random() < self.destroy_ratio:
                        destroyed[t, i] = 0

        return destroyed

    def _repair(
        self,
        destroyed: np.ndarray,
        op_idx: int,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        full_demand: np.ndarray,
        T: int,
    ) -> np.ndarray:
        """Repair the destroyed solution."""
        repaired = destroyed.copy()

        if op_idx == 0:
            # Greedy repair: BaseStock for zero entries
            S = self.cfg.demand_mean * (self.cfg.lead_time + 1)
            for t in range(T):
                for i in range(self.N):
                    if repaired[t, i] == 0:
                        position = inventory[i] + pipeline[i].sum() - backlog[i]
                        q = max(0, min(int(S - position), self.cfg.max_order))
                        repaired[t, i] = q
        elif op_idx == 1:
            # Random repair
            for t in range(T):
                for i in range(self.N):
                    if repaired[t, i] == 0:
                        repaired[t, i] = self.rng.integers(0, self.cfg.max_order + 1)
        else:
            # Demand-based repair: order proportional to forecast
            for t in range(T):
                for i in range(self.N):
                    if repaired[t, i] == 0:
                        d = full_demand[t, i] if t < full_demand.shape[0] else self.cfg.demand_mean
                        q = max(0, min(int(d * (self.cfg.lead_time + 1) - inventory[i]), self.cfg.max_order))
                        repaired[t, i] = q

        return repaired

    def _evaluate_cost(
        self,
        solution: np.ndarray,
        init_inventory: np.ndarray,
        init_backlog: np.ndarray,
        pipeline: np.ndarray,
        full_demand: np.ndarray,
    ) -> float:
        """Evaluate total cost of a solution."""
        T = solution.shape[0]
        inv = init_inventory.copy()
        bl = init_backlog.copy()
        pipe = pipeline.copy()
        total_cost = 0.0

        for t in range(T):
            incoming = pipe[:, 0]
            inv += incoming
            pipe[:, :-1] = pipe[:, 1:]
            for i in range(self.N):
                pipe[i, -1] = solution[t, i]

            demand = full_demand[t]
            for i in self.topo_order:
                total_need = demand[i] + bl[i]
                fulfilled = min(inv[i], total_need)
                inv[i] -= fulfilled
                bl[i] = total_need - fulfilled

            total_cost += (self.cfg.H * np.maximum(inv, 0)).sum() + (self.cfg.B * bl).sum()

        return total_cost

    def _expand_demand_forecast(self, demand_forecast: np.ndarray) -> np.ndarray:
        T = demand_forecast.shape[0]
        full_demand = np.zeros((T, self.N), dtype=np.float64)
        for t in range(T):
            for idx, terminal in enumerate(self.terminals):
                full_demand[t, terminal] = demand_forecast[t, idx]
            for i in range(self.N):
                if i not in set(self.terminals):
                    full_demand[t, i] = self.cfg.demand_mean * len(self.downstream[i]) / max(len(self.terminals), 1)
        return full_demand

    def _basestock_solution(self, inventory, backlog, pipeline, T):
        S = self.cfg.demand_mean * (self.cfg.lead_time + 1)
        solution = np.zeros((T, self.N), dtype=int)
        for t in range(T):
            for i in range(self.N):
                position = inventory[i] + pipeline[i].sum() - backlog[i]
                q = max(0, min(int(S - position), self.cfg.max_order))
                solution[t, i] = q
        return solution

    def get_actions(self, inventory, backlog, pipeline, current_demand_forecast):
        orders = self.solve(inventory, backlog, pipeline, current_demand_forecast)
        actions = []
        for i in range(self.cfg.num_nodes):
            q = orders[0][i] if len(orders) > 0 else 0
            k = len(self.upstream[i])
            alpha = np.array([1.0]) if k == 0 else np.full(k, 1.0 / k)
            actions.append((q, alpha))
        return actions
