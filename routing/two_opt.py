"""2-opt local search for fixed-depot routes."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .distance import Coord, route_distance


def two_opt_route(
    route: Sequence[int],
    coords: Sequence[Coord],
    max_iterations: int = 500,
    tolerance: float = 1e-9,
) -> List[int]:
    """Improve a route with 2-opt while keeping first/last nodes fixed."""
    best = list(route)
    if len(best) <= 4:
        return best

    best_dist = route_distance(best, coords)
    for _ in range(max_iterations):
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cand_dist = route_distance(candidate, coords)
                if cand_dist + tolerance < best_dist:
                    best = candidate
                    best_dist = cand_dist
                    improved = True
        if not improved:
            break
    return best
