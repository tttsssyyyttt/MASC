"""Genetic algorithm solver for inventory/order quantities."""

from __future__ import annotations

from typing import List

import numpy as np

from ..core.config import EnvConfig
from ..core.env import MEIRPEnv
from ..core.network import build_network


class GAInventorySolver:
    """Rolling-horizon GA over per-node order quantities.

    The solver returns current-project BDQ actions.  Routing, if enabled, is
    handled by `MEIRPEnv` after the action is executed.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        population_size: int = 24,
        n_generations: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_ratio: float = 0.15,
        planning_horizon: int = 5,
        seed: int = 42,
    ):
        self.cfg = cfg
        self.population_size = max(2, int(population_size))
        self.n_generations = max(1, int(n_generations))
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elite_ratio = float(elite_ratio)
        self.planning_horizon = max(1, int(planning_horizon))
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.upstream, self.downstream, self.topo_order, self.terminals = build_network(cfg)
        self.N = cfg.num_nodes
        self.L = cfg.lead_time
        self.H = np.asarray(cfg.H, dtype=np.float64)
        self.B = np.asarray(cfg.B, dtype=np.float64)
        self.last_best_cost = None
        self.last_basestock_cost = None

    def solve(
        self,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        demand_forecast: np.ndarray,
    ) -> List[List[int]]:
        """Return a rolling-horizon order plan."""
        T = min(self.planning_horizon, int(demand_forecast.shape[0]))
        full_demand = self._expand_demand_forecast(demand_forecast[:T])
        base = self._basestock_solution(inventory, backlog, pipeline, T)
        base_cost = self._evaluate_cost(base, inventory, backlog, pipeline, full_demand)

        population = self._init_population(T, base)
        best = base.copy()
        best_cost = base_cost

        for _ in range(self.n_generations):
            costs = np.array([
                self._evaluate_cost(chrom, inventory, backlog, pipeline, full_demand)
                for chrom in population
            ])
            best_idx = int(costs.argmin())
            if costs[best_idx] < best_cost:
                best_cost = float(costs[best_idx])
                best = population[best_idx].copy()
            population = self._evolve(population, costs)

        self.last_best_cost = float(best_cost)
        self.last_basestock_cost = float(base_cost)
        return [best[t].astype(int).tolist() for t in range(T)]

    def get_actions(self, env: MEIRPEnv) -> list:
        forecast = self._default_forecast(env)
        orders = self.solve(env.inventory, env.backlog, env.pipeline, forecast)
        first = orders[0] if orders else [0] * self.N
        return self._orders_to_bdq(first)

    def _init_population(self, T: int, base: np.ndarray) -> List[np.ndarray]:
        population = [base.copy()]
        while len(population) < self.population_size:
            chrom = self.rng.integers(0, self.cfg.max_order + 1, size=(T, self.N))
            population.append(chrom.astype(int))
        return population

    def _evolve(self, population: List[np.ndarray], costs: np.ndarray) -> List[np.ndarray]:
        elite_n = max(1, int(round(self.population_size * self.elite_ratio)))
        elite_idx = costs.argsort()[:elite_n]
        new_population = [population[i].copy() for i in elite_idx]

        while len(new_population) < self.population_size:
            p1 = self._tournament(population, costs)
            p2 = self._tournament(population, costs)
            if self.rng.random() < self.crossover_rate:
                c1, c2 = self._crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_population.append(self._mutate(c1))
            if len(new_population) < self.population_size:
                new_population.append(self._mutate(c2))
        return new_population[:self.population_size]

    def _tournament(self, population: List[np.ndarray], costs: np.ndarray, k: int = 3) -> np.ndarray:
        size = min(k, len(population))
        idx = self.rng.choice(len(population), size=size, replace=False)
        return population[int(idx[costs[idx].argmin()])]

    def _crossover(self, p1: np.ndarray, p2: np.ndarray):
        mask = self.rng.random(p1.shape) < 0.5
        return np.where(mask, p1, p2), np.where(mask, p2, p1)

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        child = chromosome.copy()
        mask = self.rng.random(child.shape) < self.mutation_rate
        child[mask] = self.rng.integers(0, self.cfg.max_order + 1, size=int(mask.sum()))
        return child.astype(int)

    def _evaluate_cost(
        self,
        orders: np.ndarray,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        full_demand: np.ndarray,
    ) -> float:
        inv = inventory.astype(np.float64).copy()
        bl = backlog.astype(np.float64).copy()
        pipe = pipeline.astype(np.float64).copy()
        total = 0.0

        for t in range(orders.shape[0]):
            inv += pipe[:, 0]
            pipe[:, :-1] = pipe[:, 1:]
            pipe[:, -1] = orders[t]

            demand = full_demand[t]
            for i in self.topo_order:
                need = demand[i] + bl[i]
                fulfilled = min(inv[i], need)
                inv[i] -= fulfilled
                bl[i] = need - fulfilled

            total += float((self.H * np.maximum(inv, 0)).sum() + (self.B * bl).sum())
            total += self._estimate_transport_cost(orders[t])

        return total

    def _estimate_transport_cost(self, orders_t: np.ndarray) -> float:
        if not self.cfg.routing_enabled or self.cfg.transport_cost <= 0:
            return 0.0
        return float(np.count_nonzero(orders_t) * self.cfg.transport_cost)

    def _expand_demand_forecast(self, demand_forecast: np.ndarray) -> np.ndarray:
        T = demand_forecast.shape[0]
        full = np.zeros((T, self.N), dtype=np.float64)
        terminal_set = set(self.terminals)
        for t in range(T):
            for idx, terminal in enumerate(self.terminals):
                full[t, terminal] = demand_forecast[t, idx]
            for i in reversed(self.topo_order):
                if i in terminal_set:
                    continue
                full[t, i] = sum(full[t, d] for d in self.downstream[i])
        return full

    def _basestock_solution(self, inventory, backlog, pipeline, T: int) -> np.ndarray:
        target = self.cfg.demand_mean * (self.cfg.lead_time + 1)
        solution = np.zeros((T, self.N), dtype=int)
        for t in range(T):
            for i in range(self.N):
                position = inventory[i] + pipeline[i].sum() - backlog[i]
                q = max(0, min(int(target - position), self.cfg.max_order))
                solution[t, i] = q
        return solution

    def _default_forecast(self, env: MEIRPEnv) -> np.ndarray:
        return np.full(
            (self.planning_horizon, len(env.terminals)),
            float(self.cfg.demand_mean),
            dtype=np.float64,
        )

    def _orders_to_bdq(self, orders: List[int]) -> list:
        actions = []
        for i, q_raw in enumerate(orders):
            q = max(0, min(int(q_raw), self.cfg.max_order))
            k = len(self.upstream[i])
            if k == 0:
                actions.append(q)
            else:
                share = q // k
                rem = q % k
                action_i = [share] * k
                for j in range(rem):
                    action_i[j] += 1
                actions.append(action_i)
        return actions
