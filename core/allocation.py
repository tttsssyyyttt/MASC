"""Order allocation — pure functions, zero state."""

from typing import List

import numpy as np


def allocate(q: int, alpha: List[float], upstream_avail: List[float]) -> List[float]:
    """Allocate order quantity q across upstream nodes.

    Fair Share rule:
    - If upstream has enough: actual = q * alpha_k
    - If upstream is short: scale down proportionally across all downstream requests

    Args:
        q: total order quantity
        alpha: allocation ratios per upstream (sum ≈ 1)
        upstream_avail: available inventory at each upstream node

    Returns:
        shipments: actual shipment from each upstream
    """
    k = len(alpha)
    if k == 0:
        return []

    shipments = [0.0] * k
    remaining_q = float(q)

    # First pass: try to allocate by alpha ratios
    for i in range(k):
        requested = q * alpha[i]
        actual = min(requested, upstream_avail[i])
        shipments[i] = actual
        remaining_q -= actual
        upstream_avail[i] -= actual

    # If some upstream couldn't fulfill, redistribute remaining_q
    # to upstream nodes that still have stock, proportionally
    if remaining_q > 1e-9:
        total_remaining_avail = sum(max(av, 0) for av in upstream_avail)
        if total_remaining_avail > 1e-9:
            for i in range(k):
                if upstream_avail[i] > 1e-9:
                    extra = min(
                        remaining_q * (upstream_avail[i] / total_remaining_avail),
                        upstream_avail[i],
                    )
                    shipments[i] += extra
                    remaining_q -= extra
                    upstream_avail[i] -= extra

    # Round to avoid floating drift — ensure sum <= q
    total = sum(shipments)
    if total > q:
        scale = q / total
        shipments = [s * scale for s in shipments]

    return shipments
