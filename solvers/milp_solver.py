"""MILP Solver — rolling horizon exact optimization for MEIRP.

Uses PuLP for mixed-integer linear programming.
Provides an upper bound on performance and can serve as expert advisor.

Design decision: The MILP uses aggregated order variables q[i,t] (total order
per node per step) rather than per-edge variables ord[dst, src, t]. This is
because the per-edge formulation, while more expressive, creates a rolling-
horizon inconsistency: the model at step t can overload an upstream beyond
its capacity, relying on future re-optimization that may not materialize.

The aggregated formulation avoids this by treating demand for non-terminal
nodes as a fixed quantity (sum of descendant terminal demands), which makes
each node's inventory balance independent. Alpha (allocation ratios) is then
optimized post-solve using a load-balancing LP.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..core.config import EnvConfig
from ..core.network import build_network


class MILPSolver:
    """MILP-based rolling horizon solver.

    Given current state and demand forecast, solves for optimal
    order quantities over a planning horizon.
    """

    def __init__(self, cfg: EnvConfig, planning_horizon: int = 15):
        self.cfg = cfg
        self.planning_horizon = planning_horizon
        self.upstream, self.downstream, self.topo_order, self.terminals = build_network(cfg)

    def solve(
        self,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        demand_forecast: np.ndarray,  # (T, n_terminals)
    ) -> List[List[int]]:
        """Solve MILP for optimal order quantities.

        Args:
            inventory: (N,) current inventory
            backlog: (N,) current backlog
            pipeline: (N, L) in-transit orders
            demand_forecast: (T, n_terminals) demand forecast

        Returns:
            orders: (T, N) optimal order quantities per timestep per node
        """
        try:
            import pulp
        except ImportError:
            return self._heuristic_fallback(inventory, backlog, pipeline, demand_forecast)

        cfg = self.cfg
        N = cfg.num_nodes
        T = min(self.planning_horizon, demand_forecast.shape[0])
        L = cfg.lead_time

        prob = pulp.LpProblem("MEIRP_MILP", pulp.LpMinimize)

        # Decision variables: q[i][t] = total order quantity for node i at time t
        q = {}
        for i in range(N):
            for t in range(T):
                q[i, t] = pulp.LpVariable(f"q_{i}_{t}", lowBound=0, upBound=cfg.max_order, cat="Integer")

        # State variables
        inv = {}
        bl = {}
        for i in range(N):
            for t in range(T):
                inv[i, t] = pulp.LpVariable(f"inv_{i}_{t}", lowBound=0)
                bl[i, t] = pulp.LpVariable(f"bl_{i}_{t}", lowBound=0)

        # Objective: minimize total cost
        cost = pulp.lpSum(
            cfg.H[i] * inv[i, t] + cfg.B[i] * bl[i, t]
            for i in range(N) for t in range(T)
        )
        prob += cost

        # Constraints: inventory balance
        for i in range(N):
            for t in range(T):
                # Incoming shipment
                if t < L:
                    incoming = float(pipeline[i, t]) if t < pipeline.shape[1] else 0.0
                else:
                    incoming = q[i, t - L]

                # Demand
                if i in self.terminals:
                    term_idx = self.terminals.index(i)
                    dem = float(demand_forecast[t, term_idx]) if t < demand_forecast.shape[0] else float(cfg.demand_mean)
                else:
                    # Aggregate descendant terminal demand (fixed, not variable-dependent)
                    ds_demand = self._get_descendant_demand(i, demand_forecast, t, self.terminals)
                    dem = ds_demand if ds_demand > 0 else float(cfg.demand_mean)

                # Balance
                if t == 0:
                    prob += inv[i, t] - bl[i, t] == float(inventory[i]) + incoming - float(backlog[i]) - dem
                else:
                    prob += inv[i, t] - bl[i, t] + bl[i, t - 1] == inv[i, t - 1] + incoming - dem

        # Solve (with time limit and fallback on solver error)
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
        except Exception:
            return self._heuristic_fallback(inventory, backlog, pipeline, demand_forecast)

        # Extract solution
        orders = []
        for t in range(T):
            step_orders = []
            for i in range(N):
                if q[i, t].varValue is not None:
                    step_orders.append(max(0, int(round(q[i, t].varValue))))
                else:
                    step_orders.append(0)
            orders.append(step_orders)

        return orders

    def _get_descendant_demand(self, node, demand_forecast, t, terminals):
        """Recursively compute total terminal demand for all descendants of node."""
        total = 0.0
        for d in self.downstream[node]:
            if d in terminals:
                term_idx = terminals.index(d)
                if t < demand_forecast.shape[0]:
                    total += float(demand_forecast[t, term_idx])
                else:
                    total += float(self.cfg.demand_mean)
            else:
                total += self._get_descendant_demand(d, demand_forecast, t, terminals)
        return total

    def _heuristic_fallback(
        self,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        demand_forecast: np.ndarray,
    ) -> List[List[int]]:
        """BaseStock heuristic fallback when PuLP is unavailable."""
        cfg = self.cfg
        N = cfg.num_nodes
        T = min(self.planning_horizon, demand_forecast.shape[0])
        S = cfg.demand_mean * (cfg.lead_time + 1)

        orders = []
        for t in range(T):
            step_orders = []
            for i in range(N):
                pipe_sum = pipeline[i].sum() if t == 0 else 0
                position = inventory[i] + pipe_sum - backlog[i]
                step_orders.append(max(0, min(int(S - position), cfg.max_order)))
            orders.append(step_orders)

        return orders

    def get_actions(self, inventory, backlog, pipeline, current_demand_forecast):
        """Get actions for the current step using MILP + alpha optimization.

        1. Solve aggregated MILP for optimal total q[i] per node
        2. Solve a small LP to optimally allocate q[i] among upstreams
           (minimizing max utilization / balancing load)

        Returns:
            actions: list of per-agent actions in BDQ format
                - root node: int (single order quantity)
                - non-root: list[int] (per-upstream order quantities)
        """
        orders = self.solve(inventory, backlog, pipeline, current_demand_forecast)
        q_vals = np.array([orders[0][i] if len(orders) > 0 else 0 for i in range(self.cfg.num_nodes)], dtype=np.float64)

        # Step 2: Optimize alpha (LP if possible, heuristic fallback)
        try:
            alpha = self._optimize_alpha_lp(q_vals, inventory, pipeline)
        except Exception:
            alpha = self._optimize_alpha_heuristic(q_vals, inventory, pipeline)

        # Convert to BDQ format
        actions = []
        for i in range(self.cfg.num_nodes):
            k = len(self.upstream[i])
            q_i = int(q_vals[i])
            if k == 0:
                actions.append(q_i)
            else:
                per_up = []
                for j, u in enumerate(self.upstream[i]):
                    share = int(round(alpha[i][j] * q_i))
                    per_up.append(share)
                # Adjust for rounding: add remainder to first upstream
                total_alloc = sum(per_up)
                if total_alloc < q_i and k > 0:
                    per_up[0] += q_i - total_alloc
                elif total_alloc > q_i:
                    per_up[0] -= total_alloc - q_i
                    per_up[0] = max(0, per_up[0])
                actions.append(per_up)

        return actions

    def _optimize_alpha_lp(self, q, inventory, pipeline):
        """Solve a small LP to optimally allocate q[i] among upstreams.

        Objective: minimize the maximum utilization rate across all nodes.
        A node's utilization = (committed_demand) / (inventory + pipeline + max_order).

        This balances load: nodes with more upstream capacity absorb more orders.
        """
        try:
            import pulp
        except ImportError:
            return self._optimize_alpha_heuristic(q, inventory, pipeline)

        N = self.cfg.num_nodes
        pipe_sum = pipeline.sum(axis=1)

        # Capacity per node = current inventory + pipeline + max future order
        capacity = inventory + pipe_sum + self.cfg.max_order
        capacity = np.maximum(capacity, 1.0)  # avoid div-by-zero

        prob = pulp.LpProblem("Alpha_Allocation", pulp.LpMinimize)

        # Variables: alpha[i][j] for node i, upstream j
        alpha_vars = {}
        for i in range(N):
            k = len(self.upstream[i])
            if k <= 1:
                continue
            for j in range(k):
                alpha_vars[i, j] = pulp.LpVariable(f"a_{i}_{j}", lowBound=0, upBound=1)

        # Variable: max utilization (minimax objective)
        max_util = pulp.LpVariable("max_util", lowBound=0)

        # Constraints
        for i in range(N):
            k = len(self.upstream[i])
            if k == 0:
                continue
            elif k == 1:
                # Single upstream: takes 100%
                u = self.upstream[i][0]
                # This load is already committed
                continue
            else:
                # Sum of alphas = 1
                prob += pulp.lpSum(alpha_vars[i, j] for j in range(k)) == 1.0

        # Utilization constraints per upstream
        for u in range(N):
            if len(self.downstream[u]) == 0:
                continue  # terminal, no downstream depends on it

            # Load on upstream u = sum of downstream orders allocated to u
            load_terms = []
            for d in self.downstream[u]:
                k = len(self.upstream[d])
                if k == 0:
                    continue  # root ordering from external
                elif k == 1:
                    # Exclusive: entire q[d] goes to u
                    load_terms.append(q[d])
                else:
                    # Shared: find j such that upstream[d][j] == u
                    j = list(self.upstream[d]).index(u)
                    load_terms.append(alpha_vars[d, j] * q[d])

            if load_terms:
                total_load = pulp.lpSum(load_terms)
                # Utilization = total_load / capacity[u] ≤ max_util
                prob += total_load / float(capacity[u]) <= max_util

        # Objective: minimize max utilization + small penalty for deviation from uniform
        prob += max_util

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))

        # Extract alphas
        alphas = []
        for i in range(N):
            k = len(self.upstream[i])
            if k == 0:
                alphas.append(np.array([1.0]))
            elif k == 1:
                alphas.append(np.array([1.0]))
            else:
                a = []
                for j in range(k):
                    v = alpha_vars.get((i, j))
                    a.append(v.varValue if v and v.varValue is not None else 1.0 / k)
                # Normalize
                total = sum(a)
                if total > 1e-9:
                    a = [x / total for x in a]
                else:
                    a = [1.0 / k] * k
                alphas.append(np.array(a))

        return alphas

    def _optimize_alpha_heuristic(self, q, inventory, pipeline):
        """Fallback heuristic: allocate proportional to free capacity."""
        N = self.cfg.num_nodes
        pipe_sum = pipeline.sum(axis=1)
        available = inventory + pipe_sum + self.cfg.max_order

        # Committed load: q from downstreams that have this node as ONLY upstream
        committed = np.zeros(N, dtype=np.float64)
        for i in range(N):
            ups = self.upstream[i]
            if len(ups) == 1:
                u = ups[0]
                committed[u] += q[i]

        free = np.maximum(available - committed, 1.0)

        alphas = []
        for i in range(N):
            k = len(self.upstream[i])
            if k <= 1:
                alphas.append(np.array([1.0]))
            else:
                weights = np.array([free[u] for u in self.upstream[i]], dtype=np.float64)
                total = weights.sum()
                if total > 1e-9:
                    alphas.append(weights / total)
                else:
                    alphas.append(np.full(k, 1.0 / k))

        return alphas
