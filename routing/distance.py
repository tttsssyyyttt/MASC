"""Distance helpers for routing algorithms."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

Coord = Tuple[float, float]


def euclidean(a: Coord, b: Coord) -> float:
    """Return Euclidean distance between two coordinates."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_distance(route: Sequence[int], coords: Sequence[Coord]) -> float:
    """Return total distance of a node route."""
    if len(route) < 2:
        return 0.0
    total = 0.0
    for i in range(len(route) - 1):
        total += euclidean(coords[route[i]], coords[route[i + 1]])
    return total


def default_coords(layers: Sequence[int]) -> List[Coord]:
    """Create deterministic fallback coordinates from layer ids.

    These coordinates are only a fallback for routing-enabled experiments whose
    config does not provide explicit coordinates.  They do not affect inventory
    dynamics.
    """
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for node, layer in enumerate(layers):
        by_layer[int(layer)].append(node)

    coords: List[Coord] = [(0.0, 0.0)] * len(layers)
    for layer in sorted(by_layer):
        nodes = by_layer[layer]
        center = (len(nodes) - 1) / 2.0
        for pos, node in enumerate(nodes):
            x = float(layer) * 10.0
            y = (float(pos) - center) * 5.0
            coords[node] = (x, y)
    return coords
