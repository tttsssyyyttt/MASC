"""Lightweight route result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Route:
    """One vehicle route from a depot through downstream nodes and back."""

    depot: int
    stops: Tuple[int, ...]
    loads: Tuple[float, ...]
    distance: float

    @property
    def load(self) -> float:
        return float(sum(self.loads))

    @property
    def node_sequence(self) -> Tuple[int, ...]:
        return (self.depot, *self.stops, self.depot)

    def to_dict(self) -> dict:
        return {
            "depot": self.depot,
            "stops": list(self.stops),
            "loads": list(self.loads),
            "load": self.load,
            "distance": self.distance,
            "node_sequence": list(self.node_sequence),
        }


@dataclass(frozen=True)
class RoutePlan:
    """All vehicle routes for one depot in one period."""

    depot: int
    routes: Tuple[Route, ...]
    total_distance: float
    total_load: float
    feasible: bool
    method: str

    def to_dict(self) -> dict:
        return {
            "depot": self.depot,
            "routes": [route.to_dict() for route in self.routes],
            "total_distance": self.total_distance,
            "total_load": self.total_load,
            "feasible": self.feasible,
            "method": self.method,
        }


@dataclass(frozen=True)
class RoutingSolution:
    """Route plans for all depots in one environment step."""

    plans: Dict[int, RoutePlan]
    total_distance: float
    total_load: float
    feasible: bool
    method: str

    def to_dict(self) -> dict:
        return {
            "plans": {depot: plan.to_dict() for depot, plan in self.plans.items()},
            "total_distance": self.total_distance,
            "total_load": self.total_load,
            "feasible": self.feasible,
            "method": self.method,
        }
