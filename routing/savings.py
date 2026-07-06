"""Clarke-Wright savings CVRP heuristic with optional 2-opt cleanup."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .distance import Coord, euclidean, route_distance
from .nearest_insertion import solve_nearest_insertion
from .two_opt import two_opt_route
from .types import Route, RoutePlan


def solve_savings(
    depot: int,
    customers: Sequence[int],
    coords: Sequence[Coord],
    demands: Dict[int, float],
    vehicle_capacity: float,
    vehicle_count: int,
    improve_2opt: bool = True,
) -> RoutePlan:
    """Solve a small CVRP subproblem with Clarke-Wright savings."""
    active = [c for c in customers if demands.get(c, 0.0) > 1e-9]
    if not active:
        return RoutePlan(depot, tuple(), 0.0, 0.0, True, "savings")

    if any(demands[c] > vehicle_capacity + 1e-9 for c in active):
        plan = solve_nearest_insertion(
            depot, active, coords, demands, vehicle_capacity, vehicle_count,
            improve_2opt=improve_2opt,
        )
        return RoutePlan(
            depot=plan.depot,
            routes=plan.routes,
            total_distance=plan.total_distance,
            total_load=plan.total_load,
            feasible=False,
            method="savings",
        )

    routes: Dict[int, List[int]] = {idx: [c] for idx, c in enumerate(active)}
    route_load: Dict[int, float] = {idx: float(demands[c]) for idx, c in enumerate(active)}
    cust_to_route: Dict[int, int] = {c: idx for idx, c in enumerate(active)}

    savings = []
    for i_idx, i in enumerate(active):
        for j in active[i_idx + 1:]:
            saving = (
                euclidean(coords[depot], coords[i])
                + euclidean(coords[depot], coords[j])
                - euclidean(coords[i], coords[j])
            )
            savings.append((saving, i, j))
    savings.sort(key=lambda item: (-item[0], item[1], item[2]))

    for _, i, j in savings:
        ri = cust_to_route.get(i)
        rj = cust_to_route.get(j)
        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]
        new_load = route_load[ri] + route_load[rj]
        if new_load > vehicle_capacity + 1e-9:
            continue

        merged = None
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_j[-1] == j and route_i[0] == i:
            merged = route_j + route_i
        elif route_i[0] == i and route_j[0] == j:
            merged = list(reversed(route_i)) + route_j
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + list(reversed(route_j))

        if merged is None:
            continue

        routes[ri] = merged
        route_load[ri] = new_load
        for c in route_j:
            cust_to_route[c] = ri
        del routes[rj]
        del route_load[rj]

    final_routes: List[Route] = []
    for route_customers in routes.values():
        node_route = [depot] + route_customers + [depot]
        if improve_2opt:
            node_route = two_opt_route(node_route, coords)
        stops = tuple(node_route[1:-1])
        loads = tuple(float(demands[c]) for c in stops)
        final_routes.append(Route(
            depot=depot,
            stops=stops,
            loads=loads,
            distance=route_distance(node_route, coords),
        ))

    total_distance = float(sum(route.distance for route in final_routes))
    total_load = float(sum(route.load for route in final_routes))
    feasible = all(route.load <= vehicle_capacity + 1e-9 for route in final_routes)
    feasible = feasible and len(final_routes) <= int(vehicle_count)

    return RoutePlan(
        depot=depot,
        routes=tuple(final_routes),
        total_distance=total_distance,
        total_load=total_load,
        feasible=feasible,
        method="savings",
    )
