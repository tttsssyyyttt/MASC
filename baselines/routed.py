"""Composed inventory-policy + routing-policy baselines."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence, Tuple

from ..core.config import EnvConfig
from ..core.env import MEIRPEnv
from .base import InventoryPolicy
from .base_stock import BaseStockPolicy
from .ss_policy import SSPolicy


def with_routing(
    cfg: EnvConfig,
    method: str,
    use_2opt: bool = True,
    transport_cost: Optional[float] = None,
    vehicle_capacity: Optional[float] = None,
    vehicle_count: Optional[int] = None,
    coords: Optional[Sequence[Tuple[float, float]]] = None,
) -> EnvConfig:
    """Return a copy of `cfg` with routing enabled."""
    return replace(
        cfg,
        routing_enabled=True,
        routing_method=method,
        routing_use_2opt=use_2opt,
        transport_cost=cfg.transport_cost if transport_cost is None else transport_cost,
        vehicle_capacity=cfg.vehicle_capacity if vehicle_capacity is None else vehicle_capacity,
        vehicle_count=cfg.vehicle_count if vehicle_count is None else vehicle_count,
        coords=cfg.coords if coords is None else list(coords),
    )


class RoutedInventoryPolicy(InventoryPolicy):
    """Pass-through inventory policy tagged with a routing method."""

    def __init__(self, inventory_policy: InventoryPolicy, routing_method: str):
        super().__init__(inventory_policy.cfg)
        self.inventory_policy = inventory_policy
        self.routing_method = routing_method

    def get_actions(self, env: MEIRPEnv) -> list:
        return self.inventory_policy.get_actions(env)


class RoutedBaseStockPolicy(RoutedInventoryPolicy):
    """BaseStock inventory policy evaluated in a routing-enabled env."""

    def __init__(self, cfg: EnvConfig, routing_method: str):
        super().__init__(BaseStockPolicy(cfg), routing_method)


class RoutedSSPolicy(RoutedInventoryPolicy):
    """(s,S) inventory policy evaluated in a routing-enabled env."""

    def __init__(self, cfg: EnvConfig, routing_method: str):
        super().__init__(SSPolicy(cfg), routing_method)
