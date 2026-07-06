"""GA (Genetic Algorithm) Solver for MEIRP.

Evolves a population of order quantity vectors over a planning horizon.
Uses demand forecast that is gradually revealed (rolling horizon).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.config import EnvConfig
from ..core.network import build_network


class GASolver:
    """Genetic Algorithm solver for MEIRP.

    Each chromosome encodes order quantities for all nodes over a horizon.
    Fitness = negative total cost (holding + backlog).
    """

    def __init__(
        self,
        cfg: EnvConfig,
        population_size: int = 50,
        n_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_ratio: float = 0.1,
        planning_horizon: int = 5,
    ):
        self.cfg = cfg
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.planning_horizon = planning_horizon

        self.upstream, self.downstream, self.topo_order, self.terminals = build_network(cfg)
        self.N = cfg.num_nodes
        self.L = cfg.lead_time

    def solve(
        self,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        demand_forecast: np.ndarray,
    ) -> List[List[int]]:
        """Solve using GA.

        Args:
            inventory: (N,) current inventory
            backlog: (N,) current backlog
            pipeline: (N, L) in-transit orders
            demand_forecast: (T, n_terminals) demand forecast

        Returns:
            orders: (T, N) order quantities per timestep per node
        """
        T = min(self.planning_horizon, demand_forecast.shape[0])

        # Initialize population: each chromosome = (T, N) order quantities
        population = self._init_population(T)

        # Expand demand forecast to all nodes
        full_demand = self._expand_demand_forecast(demand_forecast[:T])

        best_chromosome = None
        best_fitness = -np.inf

        for gen in range(self.n_generations):
            # Evaluate fitness
            fitnesses = np.array([
                self._evaluate_fitness(chrom, inventory, backlog, pipeline, full_demand)
                for chrom in population
            ])

            # Track best
            gen_best_idx = fitnesses.argmax()
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()

            # Selection + Crossover + Mutation
            population = self._evolve(population, fitnesses)

        if best_chromosome is None:
            # Fallback: BaseStock heuristic
            return self._heuristic_fallback(inventory, backlog, pipeline, demand_forecast)

        # Convert to list format
        orders = []
        for t in range(T):
            orders.append(best_chromosome[t].tolist())

        return orders

    def _init_population(self, T: int) -> List[np.ndarray]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            chrom = np.random.randint(0, self.cfg.max_order + 1, size=(T, self.N))
            population.append(chrom)
        return population

    def _evaluate_fitness(
        self,
        chromosome: np.ndarray,
        init_inventory: np.ndarray,
        init_backlog: np.ndarray,
        pipeline: np.ndarray,
        full_demand: np.ndarray,
    ) -> float:
        """Evaluate fitness = negative total cost.

        Simulate the supply chain with given order quantities.
        """
        T = chromosome.shape[0]
        inv = init_inventory.copy()
        bl = init_backlog.copy()
        pipe = pipeline.copy()
        total_cost = 0.0

        for t in range(T):
            # Pipeline arrival
            incoming = pipe[:, 0]
            inv += incoming

            # Shift pipeline
            pipe[:, :-1] = pipe[:, 1:]

            # New orders enter pipeline
            for i in range(self.N):
                pipe[i, -1] = chromosome[t, i]

            # Demand
            demand = full_demand[t]

            # Fulfill demand (topological order)
            for i in self.topo_order:
                total_need = demand[i] + bl[i]
                fulfilled = min(inv[i], total_need)
                inv[i] -= fulfilled
                bl[i] = total_need - fulfilled

            # Cost
            holding = self.cfg.H * np.maximum(inv, 0)
            backlog_cost = self.cfg.B * bl
            total_cost += holding.sum() + backlog_cost.sum()

        return -total_cost  # negative because we maximize fitness

    def _expand_demand_forecast(self, demand_forecast: np.ndarray) -> np.ndarray:
        """Expand terminal demand to all nodes using proportional allocation."""
        T = demand_forecast.shape[0]
        full_demand = np.zeros((T, self.N), dtype=np.float64)

        for t in range(T):
            for idx, terminal in enumerate(self.terminals):
                full_demand[t, terminal] = demand_forecast[t, idx]
            # Non-terminal demand approximation
            for i in range(self.N):
                if i not in set(self.terminals):
                    full_demand[t, i] = self.cfg.demand_mean * len(self.downstream[i]) / max(len(self.terminals), 1)

        return full_demand

    def _evolve(self, population: List[np.ndarray], fitnesses: np.ndarray) -> List[np.ndarray]:
        """Selection, crossover, and mutation."""
        n_elite = max(1, int(self.population_size * self.elite_ratio))

        # Sort by fitness (descending)
        sorted_indices = fitnesses.argsort()[::-1]

        new_population = []

        # Elitism: keep top individuals
        for i in range(n_elite):
            new_population.append(population[sorted_indices[i]].copy())

        # Tournament selection + crossover
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select(population, fitnesses)
            parent2 = self._tournament_select(population, fitnesses)

            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        return new_population[:self.population_size]

    def _tournament_select(self, population: List[np.ndarray], fitnesses: np.ndarray, k: int = 3) -> np.ndarray:
        """Tournament selection."""
        indices = np.random.choice(len(population), size=min(k, len(population)), replace=False)
        best_idx = indices[fitnesses[indices].argmax()]
        return population[best_idx]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover."""
        mask = np.random.random(parent1.shape) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Random mutation."""
        mask = np.random.random(chromosome.shape) < self.mutation_rate
        mutations = np.random.randint(0, self.cfg.max_order + 1, size=chromosome.shape)
        chromosome[mask] = mutations[mask]
        return chromosome

    def _heuristic_fallback(self, inventory, backlog, pipeline, demand_forecast):
        """BaseStock fallback."""
        T = min(self.planning_horizon, demand_forecast.shape[0])
        S = self.cfg.demand_mean * (self.cfg.lead_time + 1)
        orders = []
        for t in range(T):
            step_orders = []
            for i in range(self.N):
                pipeline_sum = pipeline[i].sum() if t == 0 else 0
                position = inventory[i] + pipeline_sum - backlog[i]
                q = max(0, min(int(S - position), self.cfg.max_order))
                step_orders.append(q)
            orders.append(step_orders)
        return orders

    def get_actions(self, inventory, backlog, pipeline, current_demand_forecast):
        """Get actions for current step using GA."""
        orders = self.solve(inventory, backlog, pipeline, current_demand_forecast)
        actions = []
        for i in range(self.cfg.num_nodes):
            q = orders[0][i] if len(orders) > 0 else 0
            k = len(self.upstream[i])
            alpha = np.array([1.0]) if k == 0 else np.full(k, 1.0 / k)
            actions.append((q, alpha))
        return actions
