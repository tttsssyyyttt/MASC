"""Integrated ALNS-style IRP solver with route-aware objective."""

from __future__ import annotations

from typing import Dict

import numpy as np

from core.config import EnvConfig
from routing.distance import default_coords
from routing.policy import RoutingPolicy
from .alns_inventory_solver import ALNSInventorySolver


class ALNSIRPSolver(ALNSInventorySolver):
    """ALNS over order quantities with a heuristic routing cost in the objective."""

    def __init__(
        self,
        cfg: EnvConfig,
        max_iterations: int = 80,
        destroy_ratio: float = 0.3,
        planning_horizon: int = 5,
        seed: int = 42,
    ):
        super().__init__(
            cfg=cfg,
            max_iterations=max_iterations,
            destroy_ratio=destroy_ratio,
            planning_horizon=planning_horizon,
            seed=seed,
        )
        self.coords = cfg.coords if cfg.coords is not None else default_coords(cfg.layers)
        method = cfg.routing_method if cfg.routing_enabled else "savings"
        self.route_policy = RoutingPolicy(method, cfg.routing_use_2opt)

    def _estimate_transport_cost(self, orders_t: np.ndarray) -> float:
        if not self.cfg.routing_enabled or self.cfg.transport_cost <= 0:
            return 0.0

        ship_detail: Dict[int, Dict[int, float]] = {i: {} for i in range(self.N)}
        for node, q_raw in enumerate(orders_t):
            q = float(max(0.0, q_raw))
            ups = self.upstream[node]
            if q <= 1e-9 or not ups:
                continue
            share = q / len(ups)
            for upstream in ups:
                ship_detail[upstream][node] = ship_detail[upstream].get(node, 0.0) + share

        solution = self.route_policy.solve(
            ship_detail=ship_detail,
            coords=self.coords,
            layers=self.cfg.layers,
            vehicle_capacity=self.cfg.vehicle_capacity,
            vehicle_count=self.cfg.vehicle_count,
        )
        penalty = 0.0 if solution.feasible else 10_000.0
        return float(solution.total_distance * self.cfg.transport_cost + penalty)
