"""Genetic algorithm CVRP routing heuristic."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .distance import Coord, route_distance
from .nearest_insertion import solve_nearest_insertion
from .two_opt import two_opt_route
from .types import Route, RoutePlan


def solve_ga_routing(
    depot: int,
    customers: Sequence[int],
    coords: Sequence[Coord],
    demands: Dict[int, float],
    vehicle_capacity: float,
    vehicle_count: int,
    population_size: int = 24,
    generations: int = 40,
    seed: int = 42,
    improve_2opt: bool = True,
) -> RoutePlan:
    """Solve a small CVRP subproblem with a permutation-based GA."""
    active = [c for c in customers if demands.get(c, 0.0) > 1e-9]
    if not active:
        return RoutePlan(depot, tuple(), 0.0, 0.0, True, "ga")

    rng = np.random.default_rng(seed)
    population_size = max(2, int(population_size))
    generations = max(1, int(generations))

    nearest = solve_nearest_insertion(
        depot, active, coords, demands, vehicle_capacity, vehicle_count,
        improve_2opt=False,
    )
    seed_perm = [node for route in nearest.routes for node in route.stops]
    seed_perm += [node for node in active if node not in seed_perm]

    population: List[List[int]] = [seed_perm]
    while len(population) < population_size:
        perm = list(active)
        rng.shuffle(perm)
        population.append(perm)

    best_perm = seed_perm
    best_cost = _permutation_cost(best_perm, depot, coords, demands, vehicle_capacity, vehicle_count)

    for _ in range(generations):
        costs = np.array([
            _permutation_cost(perm, depot, coords, demands, vehicle_capacity, vehicle_count)
            for perm in population
        ])
        idx = int(costs.argmin())
        if costs[idx] < best_cost:
            best_cost = float(costs[idx])
            best_perm = population[idx][:]

        elite_count = max(1, population_size // 5)
        elite_idx = costs.argsort()[:elite_count]
        next_population = [population[int(i)][:] for i in elite_idx]

        while len(next_population) < population_size:
            p1 = _tournament(population, costs, rng)
            p2 = _tournament(population, costs, rng)
            child = _order_crossover(p1, p2, rng)
            child = _mutate(child, rng)
            next_population.append(child)
        population = next_population

    routes = _decode(best_perm, depot, coords, demands, vehicle_capacity, improve_2opt)
    total_distance = float(sum(route.distance for route in routes))
    total_load = float(sum(route.load for route in routes))
    feasible = all(route.load <= vehicle_capacity + 1e-9 for route in routes)
    feasible = feasible and len(routes) <= int(vehicle_count)

    return RoutePlan(
        depot=depot,
        routes=tuple(routes),
        total_distance=total_distance,
        total_load=total_load,
        feasible=feasible,
        method="ga",
    )


def _decode(
    permutation: Sequence[int],
    depot: int,
    coords: Sequence[Coord],
    demands: Dict[int, float],
    vehicle_capacity: float,
    improve_2opt: bool,
) -> List[Route]:
    routes: List[Route] = []
    current: List[int] = []
    load = 0.0

    for customer in permutation:
        demand = float(demands[customer])
        if current and load + demand > vehicle_capacity + 1e-9:
            routes.append(_make_route(depot, current, coords, demands, improve_2opt))
            current = []
            load = 0.0
        current.append(customer)
        load += demand

    if current:
        routes.append(_make_route(depot, current, coords, demands, improve_2opt))
    return routes


def _make_route(
    depot: int,
    stops_in: Sequence[int],
    coords: Sequence[Coord],
    demands: Dict[int, float],
    improve_2opt: bool,
) -> Route:
    node_route = [depot] + list(stops_in) + [depot]
    if improve_2opt:
        node_route = two_opt_route(node_route, coords)
    stops = tuple(node_route[1:-1])
    return Route(
        depot=depot,
        stops=stops,
        loads=tuple(float(demands[node]) for node in stops),
        distance=route_distance(node_route, coords),
    )


def _permutation_cost(
    permutation: Sequence[int],
    depot: int,
    coords: Sequence[Coord],
    demands: Dict[int, float],
    vehicle_capacity: float,
    vehicle_count: int,
) -> float:
    routes = _decode(permutation, depot, coords, demands, vehicle_capacity, improve_2opt=False)
    distance = sum(route.distance for route in routes)
    overload = sum(max(0.0, route.load - vehicle_capacity) for route in routes)
    extra_routes = max(0, len(routes) - int(vehicle_count))
    return float(distance + 10_000.0 * overload + 10_000.0 * extra_routes)


def _tournament(population: List[List[int]], costs: np.ndarray, rng: np.random.Generator) -> List[int]:
    idx = rng.choice(len(population), size=min(3, len(population)), replace=False)
    return population[int(idx[costs[idx].argmin()])]


def _order_crossover(parent1: List[int], parent2: List[int], rng: np.random.Generator) -> List[int]:
    n = len(parent1)
    if n <= 2:
        return parent1[:]
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = [None] * n
    child[a:b + 1] = parent1[a:b + 1]
    fill = [node for node in parent2 if node not in child]
    fill_idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[fill_idx]
            fill_idx += 1
    return [int(node) for node in child]


def _mutate(permutation: List[int], rng: np.random.Generator) -> List[int]:
    child = permutation[:]
    if len(child) < 2:
        return child
    if rng.random() < 0.5:
        i, j = rng.choice(len(child), size=2, replace=False)
        child[int(i)], child[int(j)] = child[int(j)], child[int(i)]
    else:
        i, j = sorted(rng.choice(len(child), size=2, replace=False))
        child[int(i):int(j) + 1] = reversed(child[int(i):int(j) + 1])
    return child
