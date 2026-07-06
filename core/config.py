"""Configuration definitions and YAML loading for MEIRP environment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


@dataclass
class EnvConfig:
    """All parameters needed to define a supply chain environment.

    Node identity is purely integer-based. `layers[i]` gives the layer of node i,
    `edges` define the directed connectivity (upstream -> downstream).

    Example for a 1-2-4 three-layer network:
        num_nodes = 7
        layers    = [0, 1, 1, 2, 2, 2, 2]
        edges     = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    """

    # --- Topology ---
    num_nodes: int
    edges: List[Tuple[int, int]]  # (upstream, downstream)
    layers: List[int]  # layers[i] = layer index of node i

    # --- Per-node cost parameters ---
    H: List[float]  # holding cost per node
    B: List[float]  # backlog cost per node

    # --- Global parameters ---
    max_order: int = 20
    lead_time: int = 2
    episode_len: int = 30

    # --- Demand ---
    demand_type: str = "poisson"  # "poisson" / "normal" / "merton"
    demand_mean: float = 8.0
    demand_std: float = 3.0  # for normal distribution

    # --- Reward ---
    alpha: float = 0.5  # mixing: r = alpha * r_local + (1 - alpha) * r_global
    reward_scale: float = 0.01  # scale factor to keep rewards in reasonable range for RL
    stockout_penalty: float = 0.0  # extra penalty per unit of unfulfilled demand
    fill_rate_bonus: float = 0.0  # bonus per unit of fulfilled demand

    # --- Initial state ---
    init_inventory: float = 20.0
    init_pipeline: float = 8.0

    # --- Transport (0 = pure inventory) ---
    transport_coeff: float = 0.0
    routing_enabled: bool = False
    transport_cost: float = 0.0
    vehicle_capacity: float = 1_000_000.0
    vehicle_count: int = 1
    coords: Optional[List[Tuple[float, float]]] = None
    routing_method: str = "savings"
    routing_use_2opt: bool = True

    # --- Random seed ---
    seed: int = 42

    order_penalty: float = 0.0

    def __post_init__(self):
        """Validate configuration consistency."""
        n = self.num_nodes

        if len(self.layers) != n:
            raise ValueError(f"layers length {len(self.layers)} != num_nodes {n}")
        if len(self.H) != n:
            raise ValueError(f"H length {len(self.H)} != num_nodes {n}")
        if len(self.B) != n:
            raise ValueError(f"B length {len(self.B)} != num_nodes {n}")

        edge_set = set()
        for src, dst in self.edges:
            if src < 0 or src >= n or dst < 0 or dst >= n:
                raise ValueError(f"Edge ({src}, {dst}) out of range [0, {n})")
            if src == dst:
                raise ValueError(f"Self-loop detected: ({src}, {dst})")
            if (src, dst) in edge_set:
                raise ValueError(f"Duplicate edge: ({src}, {dst})")
            edge_set.add((src, dst))

        if self.lead_time < 1:
            raise ValueError(f"lead_time must be >= 1, got {self.lead_time}")
        if self.max_order < 1:
            raise ValueError(f"max_order must be >= 1, got {self.max_order}")
        if self.episode_len < 1:
            raise ValueError(f"episode_len must be >= 1, got {self.episode_len}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.transport_coeff < 0:
            raise ValueError(f"transport_coeff must be >= 0, got {self.transport_coeff}")
        if self.transport_cost < 0:
            raise ValueError(f"transport_cost must be >= 0, got {self.transport_cost}")
        if self.transport_cost == 0.0 and self.transport_coeff > 0.0:
            self.transport_cost = self.transport_coeff
        if self.vehicle_capacity <= 0:
            raise ValueError(f"vehicle_capacity must be > 0, got {self.vehicle_capacity}")
        if self.vehicle_count < 1:
            raise ValueError(f"vehicle_count must be >= 1, got {self.vehicle_count}")
        if self.coords is not None:
            if len(self.coords) != n:
                raise ValueError(f"coords length {len(self.coords)} != num_nodes {n}")
            self.coords = [tuple(map(float, coord)) for coord in self.coords]
        if self.routing_method not in {"savings", "nearest_insertion", "ga"}:
            raise ValueError(f"Unknown routing_method: {self.routing_method}")


def from_yaml(path: str) -> EnvConfig:
    """Load EnvConfig from a YAML file.

    Expected YAML structure:
        num_nodes: 7
        edges: [[0, 1], [0, 2], ...]
        layers: [0, 1, 1, 2, 2, 2, 2]
        H: [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        B: [5.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0]
        max_order: 20
        lead_time: 2
        ...
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Convert edge lists to tuples
    edges = [tuple(e) for e in raw.get("edges", [])]
    coords = raw.get("coords")
    if coords is not None:
        coords = [tuple(c) for c in coords]

    return EnvConfig(
        num_nodes=raw["num_nodes"],
        edges=edges,
        layers=raw["layers"],
        H=raw["H"],
        B=raw["B"],
        max_order=raw.get("max_order", 20),
        lead_time=raw.get("lead_time", 2),
        episode_len=raw.get("episode_len", 30),
        demand_type=raw.get("demand_type", "poisson"),
        demand_mean=raw.get("demand_mean", 8.0),
        demand_std=raw.get("demand_std", 3.0),
        alpha=raw.get("alpha", 0.5),
        reward_scale=raw.get("reward_scale", 0.01),
        stockout_penalty=raw.get("stockout_penalty", 0.0),
        fill_rate_bonus=raw.get("fill_rate_bonus", 0.0),
        init_inventory=raw.get("init_inventory", 20.0),
        init_pipeline=raw.get("init_pipeline", 8.0),
        transport_coeff=raw.get("transport_coeff", 0.0),
        routing_enabled=raw.get("routing_enabled", False),
        transport_cost=raw.get("transport_cost", 0.0),
        vehicle_capacity=raw.get("vehicle_capacity", 1_000_000.0),
        vehicle_count=raw.get("vehicle_count", 1),
        coords=coords,
        routing_method=raw.get("routing_method", "savings"),
        routing_use_2opt=raw.get("routing_use_2opt", True),
        seed=raw.get("seed", 42),
        order_penalty=raw.get("order_penalty", 0.0)
    )
