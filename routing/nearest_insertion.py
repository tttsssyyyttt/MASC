"""Nearest-insertion CVRP heuristic."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .distance import Coord, euclidean, route_distance
from .two_opt import two_opt_route
from .types import Route, RoutePlan


def solve_nearest_insertion(
    depot: int,
    customers: Sequence[int],
    coords: Sequence[Coord],
    demands: Dict[int, float],
    vehicle_capacity: float,
    vehicle_count: int,
    improve_2opt: bool = True,
) -> RoutePlan:
    """Build a CVRP plan with a nearest-insertion heuristic."""
    pending = [c for c in customers if demands.get(c, 0.0) > 1e-9]
    routes: List[Route] = []
    feasible = True

    for _ in range(max(0, int(vehicle_count))):
        if not pending:
            break

        fit = [c for c in pending if demands[c] <= vehicle_capacity + 1e-9]
        if not fit:
            feasible = False
            break

        first = min(fit, key=lambda c: euclidean(coords[depot], coords[c]))
        route = [depot, first, depot]
        route_load = float(demands[first])
        pending.remove(first)

        while pending:
            best_customer = None
            best_pos = None
            best_increase = None

            for c in pending:
                demand_c = float(demands[c])
                if route_load + demand_c > vehicle_capacity + 1e-9:
                    continue
                for pos in range(1, len(route)):
                    prev_node = route[pos - 1]
                    next_node = route[pos]
                    increase = (
                        euclidean(coords[prev_node], coords[c])
                        + euclidean(coords[c], coords[next_node])
                        - euclidean(coords[prev_node], coords[next_node])
                    )
                    if best_increase is None or increase < best_increase:
                        best_increase = increase
                        best_customer = c
                        best_pos = pos

            if best_customer is None or best_pos is None:
                break

            route.insert(best_pos, best_customer)
            route_load += float(demands[best_customer])
            pending.remove(best_customer)

        if improve_2opt:
            route = two_opt_route(route, coords)

        stops = tuple(route[1:-1])
        loads = tuple(float(demands[c]) for c in stops)
        routes.append(Route(
            depot=depot,
            stops=stops,
            loads=loads,
            distance=route_distance(route, coords),
        ))

    if pending:
        feasible = False
        # Keep a diagnostic route for remaining customers when capacity/vehicle
        # count is insufficient.  Tests and experiments should check feasible.
        for c in pending:
            route = [depot, c, depot]
            routes.append(Route(
                depot=depot,
                stops=(c,),
                loads=(float(demands[c]),),
                distance=route_distance(route, coords),
            ))

    total_distance = float(sum(route.distance for route in routes))
    total_load = float(sum(route.load for route in routes))
    feasible = feasible and all(route.load <= vehicle_capacity + 1e-9 for route in routes)
    feasible = feasible and len(routes) <= int(vehicle_count)

    return RoutePlan(
        depot=depot,
        routes=tuple(routes),
        total_distance=total_distance,
        total_load=total_load,
        feasible=feasible,
        method="nearest_insertion",
    )
