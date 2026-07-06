"""Topology derivation from EnvConfig — pure functions, zero state."""

from typing import List, Tuple

from .config import EnvConfig


def build_network(cfg: EnvConfig) -> Tuple[List[List[int]], List[List[int]], List[int], List[int]]:
    """Derive network structure from config.

    Returns:
        upstream:    upstream[i] = list of upstream node IDs for node i
        downstream:  downstream[i] = list of downstream node IDs for node i
        topo_order:  topological sort order (root first)
        terminals:   nodes with no downstream (direct customer connection)
    """
    n = cfg.num_nodes
    upstream: List[List[int]] = [[] for _ in range(n)]
    downstream: List[List[int]] = [[] for _ in range(n)]

    for src, dst in cfg.edges:
        upstream[dst].append(src)
        downstream[src].append(dst)

    # Topological sort (Kahn's algorithm)
    in_degree = [len(upstream[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order: List[int] = []

    while queue:
        # Sort for deterministic ordering
        queue.sort()
        node = queue.pop(0)
        topo_order.append(node)
        for child in downstream[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(topo_order) != n:
        raise ValueError(
            f"Cycle detected in network! Sorted {len(topo_order)} of {n} nodes."
        )

    # Terminals = nodes with no downstream
    terminals = [i for i in range(n) if len(downstream[i]) == 0]

    if not terminals:
        raise ValueError("No terminal nodes found — every node has a downstream.")

    return upstream, downstream, topo_order, terminals
