"""Routing policy adapter from shipment details to route plans."""

from __future__ import annotations

from typing import Dict, Sequence

from .distance import Coord
from .ga_routing import solve_ga_routing
from .nearest_insertion import solve_nearest_insertion
from .savings import solve_savings
from .types import RoutePlan, RoutingSolution


class RoutingPolicy:
    """Solve per-depot routing plans for one environment step."""

    def __init__(self, method: str = "savings", improve_2opt: bool = True):
        method = method.lower()
        if method not in {"savings", "nearest_insertion", "ga"}:
            raise ValueError(f"Unknown routing method: {method}")
        self.method = method
        self.improve_2opt = improve_2opt

    def solve(
        self,
        ship_detail: Dict[int, Dict[int, float]],
        coords: Sequence[Coord],
        layers: Sequence[int],
        vehicle_capacity: float,
        vehicle_count: int,
    ) -> RoutingSolution:
        """Build route plans from `ship_detail[depot][customer] = quantity`."""
        plans: Dict[int, RoutePlan] = {}

        for depot, deliveries in ship_detail.items():
            customers = [node for node, qty in deliveries.items() if qty > 1e-9]
            if not customers:
                continue
            demands = {node: float(deliveries[node]) for node in customers}

            if self.method == "ga":
                plan = solve_ga_routing(
                    depot, customers, coords, demands, vehicle_capacity,
                    vehicle_count, improve_2opt=self.improve_2opt,
                )
            elif self.method == "nearest_insertion":
                plan = solve_nearest_insertion(
                    depot, customers, coords, demands, vehicle_capacity,
                    vehicle_count, improve_2opt=self.improve_2opt,
                )
            else:
                plan = solve_savings(
                    depot, customers, coords, demands, vehicle_capacity,
                    vehicle_count, improve_2opt=self.improve_2opt,
                )
            plans[depot] = plan

        total_distance = float(sum(plan.total_distance for plan in plans.values()))
        total_load = float(sum(plan.total_load for plan in plans.values()))
        feasible = all(plan.feasible for plan in plans.values())

        return RoutingSolution(
            plans=plans,
            total_distance=total_distance,
            total_load=total_load,
            feasible=feasible,
            method=self.method,
        )
