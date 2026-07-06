"""CP-SAT Oracle Solver — full-episode global optimization for MEIRP.

Uses OR-Tools CP-SAT with per-edge decision variables.
Solves all T steps in a single optimization — no rolling horizon.
Provides TRUE optimal given perfect demand information (oracle upper bound).

Scaling: all state variables (inv, bl) and constraints use SCALE=100
to convert float costs to integers. Order variables (ord, root_q) are
kept in NATURAL units (0..max_order) and multiplied by SCALE when used
in inventory balance equations.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..core.config import EnvConfig
from ..core.network import build_network

SCALE = 100


def _sc(x: float) -> int:
    return int(round(x * SCALE))


class CPSATSolver:
    """Full-episode CP-SAT oracle solver with per-edge order variables."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.upstream, self.downstream, self.topo_order, self.terminals = build_network(cfg)
        self.N = cfg.num_nodes
        self.L = cfg.lead_time
        self.T = cfg.episode_len
        self.M = cfg.max_order

    def solve_full_episode(
        self,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        demand_sequence: np.ndarray,  # (T, n_terminals) TRUE demands (natural units)
    ) -> List[List]:
        """Solve full episode in one optimization.

        Returns:
            bdq_actions: list of per-step BDQ-format actions, length T
        """
        from ortools.sat.python import cp_model

        cfg = self.cfg
        N = self.N
        T = self.T
        L = self.L
        M = self.M

        H_int = [_sc(cfg.H[i]) for i in range(N)]
        B_int = [_sc(cfg.B[i]) for i in range(N)]
        BIG = _sc(10000.0)  # large bound for inv/bl

        model = cp_model.CpModel()

        # ============================================================
        # Decision variables (NATURAL units: 0..max_order)
        # ============================================================
        ord_vars = {}  # ord[i,u,t]: node i orders from upstream u at time t
        for i in range(N):
            for u in self.upstream[i]:
                for t in range(T):
                    ord_vars[i, u, t] = model.NewIntVar(0, M, f"ord_{i}_{u}_{t}")

        root_q = {}  # root_q[i,t]: root i orders from external supplier
        for i in range(N):
            if len(self.upstream[i]) == 0:
                for t in range(T):
                    root_q[i, t] = model.NewIntVar(0, M, f"root_q_{i}_{t}")

        # State variables (SCALED units)
        inv = {}
        bl = {}
        for i in range(N):
            for t in range(T):
                inv[i, t] = model.NewIntVar(0, BIG, f"inv_{i}_{t}")
                bl[i, t] = model.NewIntVar(0, BIG, f"bl_{i}_{t}")

        # ============================================================
        # Helpers (all return SCALED expressions)
        # ============================================================
        def total_order_scaled(i, t):
            """Total order quantity for node i at time t (SCALED)."""
            if len(self.upstream[i]) == 0:
                v = root_q.get((i, t))
                return v * SCALE if v is not None else 0
            return sum(ord_vars[i, u, t] for u in self.upstream[i]) * SCALE

        def incoming_scaled(i, t):
            """Incoming shipment to node i at time t (SCALED)."""
            if t < L:
                return _sc(float(pipeline[i, t])) if t < pipeline.shape[1] else 0
            return total_order_scaled(i, t - L)

        # ============================================================
        # Constraints
        # ============================================================
        for i in range(N):
            for t in range(T):
                inc = incoming_scaled(i, t)

                # Demand (SCALED)
                if i in self.terminals:
                    term_idx = self.terminals.index(i)
                    dem = _sc(float(demand_sequence[t, term_idx])) if t < demand_sequence.shape[0] else _sc(float(cfg.demand_mean))
                else:
                    # Non-terminal demand = sum of what downstreams order FROM i (scale it)
                    raw_sum = sum(ord_vars.get((d, i, t), 0) for d in self.downstream[i])
                    dem = raw_sum * SCALE

                # Inventory balance (all SCALED)
                if t == 0:
                    rhs = _sc(float(inventory[i])) + inc - _sc(float(backlog[i])) - dem
                    model.Add(inv[i, t] - bl[i, t] == rhs)
                else:
                    model.Add(inv[i, t] - bl[i, t] + bl[i, t - 1] == inv[i, t - 1] + inc - dem)

        # Total order cap per node per step (NATURAL units)
        for i in range(N):
            for t in range(T):
                if len(self.upstream[i]) > 0:
                    model.Add(sum(ord_vars[i, u, t] for u in self.upstream[i]) <= M)

        # ============================================================
        # Objective (minimize scaled holding + backlog costs)
        # ============================================================
        obj_terms = []
        for i in range(N):
            for t in range(T):
                obj_terms.append(H_int[i] * inv[i, t])
                obj_terms.append(B_int[i] * bl[i, t])
        model.Minimize(sum(obj_terms))

        # ============================================================
        # Solve
        # ============================================================
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._heuristic_episode(inventory, backlog, pipeline, demand_sequence)

        # ============================================================
        # Extract BDQ actions (NATURAL units)
        # ============================================================
        all_actions = []
        for t in range(T):
            step_actions = []
            for i in range(N):
                ups = self.upstream[i]
                k = len(ups)
                if k == 0:
                    v = root_q.get((i, t))
                    q = solver.Value(v) if v is not None else 0
                    step_actions.append(int(q))
                else:
                    per_up = []
                    for u in ups:
                        v = ord_vars.get((i, u, t))
                        per_up.append(int(solver.Value(v)) if v is not None else 0)
                    # Ensure total ≤ max_order (should already hold, but safety clamp)
                    total = sum(per_up)
                    if total > M:
                        scale = M / total
                        per_up = [max(0, int(x * scale)) for x in per_up]
                    step_actions.append(per_up)
            all_actions.append(step_actions)

        return all_actions

    def get_action(self, inventory, backlog, pipeline, demand_sequence=None):
        """Get BDQ actions for current step. Compatible with rolling-horizon interface."""
        if demand_sequence is None:
            n_terms = len(self.terminals)
            demand_sequence = np.random.poisson(
                self.cfg.demand_mean, size=(self.T, n_terms)
            ).astype(float)

        all_actions = self.solve_full_episode(inventory, backlog, pipeline, demand_sequence)
        return all_actions[0] if all_actions else self._heuristic_single(inventory, backlog, pipeline)

    # ================================================================
    # Fallbacks
    # ================================================================
    def _heuristic_episode(self, inventory, backlog, pipeline, demand_sequence):
        """BaseStock heuristic for full episode."""
        cfg = self.cfg
        N, T, L = self.N, self.T, self.L
        S = cfg.demand_mean * (L + 1)
        cur_inv, cur_bl = inventory.copy(), backlog.copy()
        cur_pipe = pipeline.copy()

        all_actions = []
        for t in range(T):
            step = []
            for i in range(N):
                pos = cur_inv[i] + cur_pipe[i].sum() - cur_bl[i]
                q = max(0, min(int(S - pos), self.M))
                k = len(self.upstream[i])
                if k == 0:
                    step.append(q)
                else:
                    step.append([q // k + (1 if j < q % k else 0) for j in range(k)])
            all_actions.append(step)

            # Approximate state update for heuristic
            for i in range(N):
                cur_inv[i] += cur_pipe[i, 0] if L > 0 else 0
                if i in self.terminals:
                    term_idx = self.terminals.index(i)
                    dem = demand_sequence[t, term_idx] if t < demand_sequence.shape[0] else cfg.demand_mean
                    cur_inv[i] -= dem
            cur_pipe[:, :-1] = cur_pipe[:, 1:] if L > 1 else np.zeros((N, 0))
            if L > 0:
                for i in range(N):
                    a = step[i]
                    cur_pipe[i, -1] = a if isinstance(a, (int, float)) else sum(a)
        return all_actions

    def _heuristic_single(self, inventory, backlog, pipeline):
        """BaseStock heuristic for single step."""
        cfg = self.cfg
        N = self.N
        S = cfg.demand_mean * (cfg.lead_time + 1)
        actions = []
        for i in range(N):
            pos = inventory[i] + pipeline[i].sum() - backlog[i]
            q = max(0, min(int(S - pos), self.M))
            k = len(self.upstream[i])
            if k == 0:
                actions.append(q)
            else:
                actions.append([q // k + (1 if j < q % k else 0) for j in range(k)])
        return actions
