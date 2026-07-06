"""OR-Tools CP-SAT rolling-horizon solver for MEIRP order decisions."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.config import EnvConfig
from ..core.env import MEIRPEnv
from ..core.network import build_network
from ..routing.distance import default_coords, euclidean

SCALE = 100


class ORToolsIRPSolver:
    """CP-SAT solver over rolling-horizon order quantities.

    The CP-SAT model optimizes inventory/backlog balance exactly under the
    current project schema.  Routing cost is represented as a linear distance
    proxy in the solver objective; the environment still computes the evaluated
    heuristic routes after actions are executed.
    """

    def __init__(self, cfg: EnvConfig, planning_horizon: int = 5, time_limit: float = 5.0):
        self.cfg = cfg
        self.planning_horizon = max(1, int(planning_horizon))
        self.time_limit = float(time_limit)
        self.upstream, self.downstream, self.topo_order, self.terminals = build_network(cfg)
        self.N = cfg.num_nodes
        self.L = cfg.lead_time
        self.M = cfg.max_order
        self.coords = cfg.coords if cfg.coords is not None else default_coords(cfg.layers)
        self.last_status = None
        self.last_objective = None
        self.last_basestock_objective = None

    def solve(
        self,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        demand_forecast: np.ndarray,
    ) -> List[List[int]]:
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            return self._basestock_plan(inventory, backlog, pipeline, demand_forecast)

        T = min(self.planning_horizon, int(demand_forecast.shape[0]))
        model = cp_model.CpModel()
        big = int(max(10_000, self.cfg.demand_mean * self.cfg.episode_len * 10) * SCALE)

        q = {
            (i, t): model.NewIntVar(0, self.M, f"q_{i}_{t}")
            for i in range(self.N) for t in range(T)
        }
        inv = {
            (i, t): model.NewIntVar(0, big, f"inv_{i}_{t}")
            for i in range(self.N) for t in range(T)
        }
        bl = {
            (i, t): model.NewIntVar(0, big, f"bl_{i}_{t}")
            for i in range(self.N) for t in range(T)
        }

        terminal_index = {node: idx for idx, node in enumerate(self.terminals)}

        def incoming_scaled(i: int, t: int):
            if t < self.L:
                if t < pipeline.shape[1]:
                    return int(round(float(pipeline[i, t]) * SCALE))
                return 0
            return q[i, t - self.L] * SCALE

        def demand_scaled(i: int, t: int):
            if i in terminal_index:
                idx = terminal_index[i]
                if t < demand_forecast.shape[0]:
                    return int(round(float(demand_forecast[t, idx]) * SCALE))
                return int(round(float(self.cfg.demand_mean) * SCALE))
            return sum(q[d, t] for d in self.downstream[i]) * SCALE

        for i in range(self.N):
            for t in range(T):
                inc = incoming_scaled(i, t)
                dem = demand_scaled(i, t)
                if t == 0:
                    rhs = (
                        int(round(float(inventory[i]) * SCALE))
                        + inc
                        - int(round(float(backlog[i]) * SCALE))
                        - dem
                    )
                    model.Add(inv[i, t] - bl[i, t] == rhs)
                else:
                    model.Add(inv[i, t] - bl[i, t] + bl[i, t - 1] == inv[i, t - 1] + inc - dem)

        obj_terms = []
        for i in range(self.N):
            h = int(round(float(self.cfg.H[i]) * SCALE))
            b = int(round(float(self.cfg.B[i]) * SCALE))
            transport = int(round(self._transport_unit_cost(i) * SCALE))
            for t in range(T):
                obj_terms.append(h * inv[i, t])
                obj_terms.append(b * bl[i, t])
                if transport > 0:
                    obj_terms.append(transport * q[i, t])
        model.Minimize(sum(obj_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        self.last_status = solver.StatusName(status)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            plan = self._basestock_plan(inventory, backlog, pipeline, demand_forecast)
            self.last_objective = None
            self.last_basestock_objective = self._internal_cost(plan, inventory, backlog, pipeline, demand_forecast)
            return plan

        plan = []
        for t in range(T):
            plan.append([int(solver.Value(q[i, t])) for i in range(self.N)])

        base = self._basestock_plan(inventory, backlog, pipeline, demand_forecast)
        self.last_objective = self._internal_cost(plan, inventory, backlog, pipeline, demand_forecast)
        self.last_basestock_objective = self._internal_cost(base, inventory, backlog, pipeline, demand_forecast)
        return plan

    def get_actions(self, env: MEIRPEnv) -> list:
        forecast = np.full(
            (self.planning_horizon, len(env.terminals)),
            float(self.cfg.demand_mean),
            dtype=np.float64,
        )
        plan = self.solve(env.inventory, env.backlog, env.pipeline, forecast)
        return self._orders_to_bdq(plan[0] if plan else [0] * self.N)

    def get_action(
        self,
        inventory: np.ndarray,
        backlog: np.ndarray,
        pipeline: np.ndarray,
        demand_forecast: Optional[np.ndarray] = None,
    ) -> list:
        if demand_forecast is None:
            demand_forecast = np.full(
                (self.planning_horizon, len(self.terminals)),
                float(self.cfg.demand_mean),
                dtype=np.float64,
            )
        plan = self.solve(inventory, backlog, pipeline, demand_forecast)
        return self._orders_to_bdq(plan[0] if plan else [0] * self.N)

    def _orders_to_bdq(self, orders: List[int]) -> list:
        actions = []
        for i, q_raw in enumerate(orders):
            q_i = max(0, min(int(q_raw), self.M))
            k = len(self.upstream[i])
            if k == 0:
                actions.append(q_i)
            else:
                share = q_i // k
                rem = q_i % k
                action_i = [share] * k
                for j in range(rem):
                    action_i[j] += 1
                actions.append(action_i)
        return actions

    def _transport_unit_cost(self, node: int) -> float:
        if not self.cfg.routing_enabled or self.cfg.transport_cost <= 0 or not self.upstream[node]:
            return 0.0
        return min(euclidean(self.coords[u], self.coords[node]) for u in self.upstream[node]) * self.cfg.transport_cost

    def _basestock_plan(self, inventory, backlog, pipeline, demand_forecast) -> List[List[int]]:
        T = min(self.planning_horizon, int(demand_forecast.shape[0]))
        target = self.cfg.demand_mean * (self.cfg.lead_time + 1)
        plan = []
        for _ in range(T):
            step = []
            for i in range(self.N):
                position = inventory[i] + pipeline[i].sum() - backlog[i]
                step.append(max(0, min(int(target - position), self.M)))
            plan.append(step)
        return plan

    def _internal_cost(self, plan, inventory, backlog, pipeline, demand_forecast) -> float:
        inv = inventory.astype(np.float64).copy()
        bl = backlog.astype(np.float64).copy()
        pipe = pipeline.astype(np.float64).copy()
        T = min(len(plan), int(demand_forecast.shape[0]))
        terminal_index = {node: idx for idx, node in enumerate(self.terminals)}
        total = 0.0

        for t in range(T):
            orders = np.asarray(plan[t], dtype=np.float64)
            inv += pipe[:, 0]
            pipe[:, :-1] = pipe[:, 1:]
            pipe[:, -1] = orders

            demand = np.zeros(self.N, dtype=np.float64)
            for i in range(self.N):
                if i in terminal_index:
                    demand[i] = demand_forecast[t, terminal_index[i]]
                else:
                    demand[i] = sum(orders[d] for d in self.downstream[i])

            for i in self.topo_order:
                need = demand[i] + bl[i]
                fulfilled = min(inv[i], need)
                inv[i] -= fulfilled
                bl[i] = need - fulfilled

            total += float(np.dot(self.cfg.H, np.maximum(inv, 0.0)) + np.dot(self.cfg.B, bl))
            for i in range(self.N):
                total += self._transport_unit_cost(i) * orders[i]
        return total
