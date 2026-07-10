"""MEIRP Supply Chain Environment - Gymnasium interface."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from numpy.random import Generator

from .allocation import allocate
from .config import EnvConfig
from .demand import DemandGenerator
from .network import build_network
from .reward import compute_reward
from routing.distance import default_coords
from routing.policy import RoutingPolicy
from routing.types import RoutingSolution


class MEIRPEnv(gym.Env):
    """Multi-Echelon Inventory Routing Problem environment.

    Observation per agent:
        [inventory, backlog, demand, requested_demand, external_demand, layer,
         pipeline(L), demand_hist(L), order_hist(L)]

    Action per agent:
        - root node: int q
        - non-root node: list/array [q_up0, q_up1, ...]
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig, rng: Optional[Generator] = None):
        super().__init__()
        self.cfg = cfg

        self.upstream, self.downstream, self.topo_order, self.terminals = build_network(cfg)
        self.n = cfg.num_nodes
        self.n_terminals = len(self.terminals)
        self.L = cfg.lead_time
        self.terminal_set = set(self.terminals)

        self.rng = rng or np.random.default_rng(cfg.seed)
        self.demand_gen = DemandGenerator(cfg, self.rng)

        self.inventory = np.zeros(self.n, dtype=np.float64)
        self.backlog = np.zeros(self.n, dtype=np.float64)

        self.demand = np.zeros(self.n, dtype=np.float64)  # 唯一服务需求口径
        self.need = np.zeros(self.n, dtype=np.float64)  # demand + old_backlog
        self.fulfilled_demand = np.zeros(self.n, dtype=np.float64)  # 实际满足 need 的量
        self.current_fulfilled = np.zeros(self.n, dtype=np.float64)  # 本期 demand 被满足的量

        self.external_demand = np.zeros(self.n, dtype=np.float64)
        self.downstream_order_demand = np.zeros(self.n, dtype=np.float64)

        # legacy / diagnostic only
        self.shipped_demand = np.zeros(self.n, dtype=np.float64)  # 实际发货流量，仅日志
        self.inventory_demand = self.demand  # 不再独立使用

        self.pipeline = np.zeros((self.n, self.L), dtype=np.float64)

        self.demand_hist = np.zeros((self.n, self.L), dtype=np.float64)
        self.order_hist = np.zeros((self.n, self.L), dtype=np.float64)
        self.step_num = 0

        # Original observation dim:
        # obs_dim = 3 + 3 * self.L
        obs_dim = 6 + 3 * self.L
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n, obs_dim), dtype=np.float64
        )
        self.action_space = gym.spaces.Tuple([
            gym.spaces.Discrete(cfg.max_order + 1)
            if len(self.upstream[i]) == 0
            else gym.spaces.MultiDiscrete([cfg.max_order + 1] * len(self.upstream[i]))
            for i in range(self.n)
        ])

        self.H_arr = np.array(cfg.H, dtype=np.float64)
        self.B_arr = np.array(cfg.B, dtype=np.float64)
        self.coords = cfg.coords if cfg.coords is not None else default_coords(cfg.layers)
        self.routing_policy = (
            RoutingPolicy(cfg.routing_method, cfg.routing_use_2opt)
            if cfg.routing_enabled else None
        )

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.demand_gen = DemandGenerator(self.cfg, self.rng)

        self.demand_gen.reset_step()
        self._demand_sequence = None
        self._demand_step = 0
        if options is not None and "demand_sequence" in options:
            self._demand_sequence = np.asarray(options["demand_sequence"], dtype=np.float64)
            self._demand_step = 0

        self.inventory[:] = self.cfg.init_inventory
        self.backlog[:] = 0.0

        self.demand[:] = 0.0
        self.need[:] = 0.0
        self.fulfilled_demand[:] = 0.0
        self.current_fulfilled[:] = 0.0

        self.external_demand[:] = 0.0
        self.downstream_order_demand[:] = 0.0

        self.shipped_demand[:] = 0.0
        self.inventory_demand = self.demand

        self.fulfilled_demand[:] = 0.0
        self.pipeline[:] = self.cfg.init_pipeline
        self.demand_hist[:] = 0.0
        self.order_hist[:] = 0.0
        self.step_num = 0

        return self._get_obs(), {}

    def _parse_actions(self, actions):
        """Parse BDQ actions into total order quantity and allocation ratios."""
        order_qty = np.zeros(self.n, dtype=np.float64)
        alpha_per_node: Dict[int, List[float]] = {}

        for i in range(self.n):
            ups = self.upstream[i]
            k = len(ups)

            if k == 0:
                if isinstance(actions[i], dict):
                    q = int(actions[i].get("q", 0))
                else:
                    q = int(actions[i])
                q = max(0, min(q, self.cfg.max_order))
                order_qty[i] = q
                alpha_per_node[i] = []
                continue

            if isinstance(actions[i], dict):
                total_q = float(max(0, min(int(actions[i].get("q", 0)), self.cfg.max_order)))
                alpha_raw = np.array(actions[i].get("alpha", []), dtype=np.float64)
                if len(alpha_raw) != k:
                    alpha_raw = np.ones(k, dtype=np.float64)
                alpha_raw = np.maximum(alpha_raw, 0.0)
                if alpha_raw.sum() > 1e-9:
                    q_per_up = total_q * alpha_raw / alpha_raw.sum()
                else:
                    q_per_up = np.full(k, total_q / k)
            else:
                q_per_up = np.array(actions[i], dtype=np.float64)
                total_q = q_per_up.sum()
                if total_q > self.cfg.max_order and total_q > 1e-9:
                    scale = self.cfg.max_order / total_q
                    q_per_up = q_per_up * scale
                    total_q = float(self.cfg.max_order)

            order_qty[i] = total_q
            if total_q > 1e-9:
                alpha_per_node[i] = (q_per_up / total_q).tolist()
            else:
                alpha_per_node[i] = [1.0 / k] * k

        return order_qty, alpha_per_node

    def _build_shipments_from_fulfilled(
            self,
            order_qty: np.ndarray,
            alpha_per_node: Dict[int, List[float]],
    ) -> Tuple[np.ndarray, Dict[int, Dict[int, float]]]:
        """Build physical shipments from fulfilled service demand.

        shipment_to[j] is inbound quantity that will arrive to node j after lead time.
        ship_detail[u][d] is physical shipment from upstream u to downstream d.
        """
        shipment_to = np.zeros(self.n, dtype=np.float64)
        ship_detail: Dict[int, Dict[int, float]] = {i: {} for i in range(self.n)}

        # 1. root 节点的 action 表示向外部供应源订货，进入 root pipeline
        for i in range(self.n):
            if len(self.upstream[i]) == 0:
                shipment_to[i] += float(int(round(order_qty[i])))

        # 2. 收集每个 upstream 被哪些 downstream 请求
        requests_by_upstream: Dict[int, List[Tuple[int, float]]] = {
            i: [] for i in range(self.n)
        }

        for d in range(self.n):
            ups = self.upstream[d]
            if not ups:
                continue

            alpha = alpha_per_node.get(d, [])
            if not alpha or sum(alpha) < 1e-9:
                alpha = [1.0 / len(ups)] * len(ups)

            alpha_sum = sum(alpha)
            total_q = float(order_qty[d])

            for j, u in enumerate(ups):
                req = total_q * alpha[j] / alpha_sum
                if req > 1e-9:
                    requests_by_upstream[u].append((d, req))

        # 3. 对每个 upstream，把它本期真正 fulfilled 的量分配给请求它的 downstream
        for u, reqs in requests_by_upstream.items():
            if not reqs:
                continue

            total_requested = sum(req for _, req in reqs)
            if total_requested <= 1e-9:
                continue

            outbound_cap = min(float(self.current_fulfilled[u]), total_requested)

            for d, req in reqs:
                amt = outbound_cap * req / total_requested
                ship_detail[u][d] = amt
                shipment_to[d] += amt

        self.shipped_demand = np.zeros(self.n, dtype=np.float64)
        for u, children in ship_detail.items():
            self.shipped_demand[u] = sum(children.values())

        # terminal 的 shipped_demand 可以用 current_fulfilled 表示对外部客户的实际服务
        for t in self.terminals:
            self.shipped_demand[t] = self.current_fulfilled[t]

        return shipment_to, ship_detail

    def _next_terminal_demands(self) -> np.ndarray:
        """Return terminal demand for the current step."""
        if self._demand_sequence is not None and self._demand_step < len(self._demand_sequence):
            terminal_demands = self._demand_sequence[self._demand_step].astype(np.float64)
            self._demand_step += 1
            return terminal_demands
        return self.demand_gen.generate(self.n_terminals)

    def _external_demand_array(self, terminal_demands: np.ndarray) -> np.ndarray:
        """Expand terminal external demand to all nodes."""
        demand_arr = np.zeros(self.n, dtype=np.float64)
        for idx, t in enumerate(self.terminals):
            demand_arr[t] = terminal_demands[idx]
        return demand_arr

    def _build_downstream_order_demand(
        self,
        order_qty: np.ndarray,
        alpha_per_node: Dict[int, List[float]],
        external_demand: np.ndarray,
    ) -> np.ndarray:
        """Service demand for each node: terminal external demand + downstream replenishment requests."""
        demand_arr = external_demand.copy()

        for d in range(self.n):
            ups = self.upstream[d]
            if not ups:
                continue
            alpha = alpha_per_node.get(d, [])
            if not alpha or sum(alpha) < 1e-9:
                alpha = [1.0 / len(ups)] * len(ups)
            alpha_sum = sum(alpha)
            for j, u in enumerate(ups):
                demand_arr[u] += order_qty[d] * alpha[j] / alpha_sum

        return demand_arr

    def _build_shipped_demand(
        self,
        ship_detail: Dict[int, Dict[int, float]],
        external_demand: np.ndarray,
    ) -> np.ndarray:
        """Demand induced by actual shipments after allocation limits."""
        demand_arr = external_demand.copy()

        for i in range(self.n):
            if i in self.terminal_set:
                continue
            total = 0.0
            for d in self.downstream[i]:
                total += ship_detail[i].get(d, 0.0)
            demand_arr[i] = total

        return demand_arr

    def _update_inventory_and_backlog(self):
        """Receive arrivals, fulfill demand, and update aggregate backlog."""
        # 1. pipeline 到货先入库
        self.inventory += self.pipeline[:, 0]

        # 2. 保存 fulfill 前的总需求，避免 fill_rate 用更新后的 backlog 反推
        old_backlog = self.backlog.copy()
        self.need = self.demand + old_backlog

        # 3. 用库存满足 need
        self.fulfilled_demand = np.minimum(self.inventory, self.need)

        # 4. 库存扣减，backlog 更新
        self.inventory -= self.fulfilled_demand
        self.backlog = self.need - self.fulfilled_demand

        # 5. 本期 demand 被满足的部分，用于 service fill / reward / 当前发货分配
        self.current_fulfilled = np.minimum(self.fulfilled_demand, self.demand)

    def _advance_pipeline(self, shipment_to: np.ndarray):
        """Shift in-transit stock and enqueue this step's shipments."""
        for i in range(self.n):
            self.pipeline[i, :-1] = self.pipeline[i, 1:]
            self.pipeline[i, -1] = shipment_to[i]

    def _record_history(self, order_qty: np.ndarray):
        """Record demand and order histories used in observations."""
        self.demand_hist[:, :-1] = self.demand_hist[:, 1:]
        self.demand_hist[:, -1] = self.demand

        self.order_hist[:, :-1] = self.order_hist[:, 1:]
        self.order_hist[:, -1] = order_qty

    def _compute_rewards(self, order_qty: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute scaled rewards and cost components."""
        cfg = self.cfg

        rewards, holding_costs, backlog_costs = compute_reward(
            self.inventory,
            self.backlog,
            self.demand,
            self.H_arr,
            self.B_arr,
            cfg.alpha,
            stockout_penalty=cfg.stockout_penalty,
            fill_rate_bonus=cfg.fill_rate_bonus,
            order_penalty=cfg.order_penalty,
            service_level_target=cfg.service_level_target,
            service_level_penalty=cfg.service_level_penalty,
            order_qty=order_qty,
            fulfilled_demand=self.current_fulfilled,
        )
        return rewards * cfg.reward_scale, holding_costs, backlog_costs

    def _apply_transport_reward(
        self,
        rewards: np.ndarray,
        transport_costs: np.ndarray,
    ) -> np.ndarray:
        """Subtract transport costs using the same local/global reward mix."""
        if not self.cfg.routing_enabled or self.cfg.transport_cost <= 0:
            return rewards
        if not np.any(transport_costs > 0):
            return rewards

        r_local = -transport_costs
        r_global = np.full(self.n, r_local.mean())
        transport_reward = self.cfg.alpha * r_local + (1 - self.cfg.alpha) * r_global
        return rewards + transport_reward * self.cfg.reward_scale

    def _solve_routing(
        self,
        ship_detail: Dict[int, Dict[int, float]],
    ) -> Tuple[Optional[RoutingSolution], np.ndarray]:
        """Solve routing for this step without changing inventory state."""
        transport_costs = np.zeros(self.n, dtype=np.float64)
        if self.routing_policy is None:
            return None, transport_costs

        solution = self.routing_policy.solve(
            ship_detail=ship_detail,
            coords=self.coords,
            layers=self.cfg.layers,
            vehicle_capacity=self.cfg.vehicle_capacity,
            vehicle_count=self.cfg.vehicle_count,
        )

        for depot, plan in solution.plans.items():
            transport_costs[depot] = plan.total_distance * self.cfg.transport_cost
        return solution, transport_costs

    def _compute_fill_rates(self) -> np.ndarray:
        """Current-period service fill rate per node."""
        fill_rates = np.ones(self.n, dtype=np.float64)
        mask = self.demand > 1e-9
        fill_rates[mask] = self.current_fulfilled[mask] / self.demand[mask]
        return np.clip(fill_rates, 0.0, 1.0)

    def _build_info(
        self,
        holding_costs: np.ndarray,
        backlog_costs: np.ndarray,
        fill_rates: np.ndarray,
        order_qty: np.ndarray,
        shipment_to: np.ndarray,
        transport_costs: np.ndarray,
        routing_solution: Optional[RoutingSolution],
    ) -> dict:
        """Build the info dictionary returned by step."""
        info = {
            "holding_costs": holding_costs,
            "backlog_costs": backlog_costs,
            "fill_rates": fill_rates,

            "order_qty": order_qty.copy(),
            "inventory": self.inventory.copy(),
            "backlog": self.backlog.copy(),

            # canonical fields
            "demand": self.demand.copy(),
            "need": self.need.copy(),
            "fulfilled_demand": self.current_fulfilled.copy(),

            # diagnostic / backward-compatible fields
            "downstream_order_demand": self.downstream_order_demand.copy(),
            "external_demand": self.external_demand.copy(),
            "inventory_demand": self.demand.copy(),
            "shipped_demand": self.shipped_demand.copy(),

            "inbound_shipment": shipment_to.copy(),
            "transport_costs": transport_costs.copy(),
            "total_transport_cost": float(transport_costs.sum()),
        }
        if routing_solution is not None:
            info.update({
                "routes": routing_solution.to_dict(),
                "route_distance": float(routing_solution.total_distance),
                "routing_feasible": bool(routing_solution.feasible),
            })
        else:
            info.update({
                "routes": None,
                "route_distance": 0.0,
                "routing_feasible": True,
            })
        return info

    def step(self, actions):
        """Execute one step."""
        # 1. 解析 action：节点向上游下多少单
        order_qty, alpha_per_node = self._parse_actions(actions)

        # 2. 生成本期 terminal 外部需求
        terminal_demands = self._next_terminal_demands()
        self.external_demand = self._external_demand_array(terminal_demands)

        # 3. 构造统一 demand：terminal external + downstream replenishment requests
        self.downstream_order_demand = self._build_downstream_order_demand(
            order_qty, alpha_per_node, self.external_demand,
        )
        self.demand = self.downstream_order_demand.copy()
        self.inventory_demand = self.demand  # legacy alias only

        # 4. pipeline 到货、满足 demand、更新 backlog
        self._update_inventory_and_backlog()

        # 5. 根据 fulfilled 生成真实 shipment 和 routing input
        shipment_to, ship_detail = self._build_shipments_from_fulfilled(
            order_qty, alpha_per_node
        )

        # 6. routing 基于真实 shipment
        routing_solution, transport_costs = self._solve_routing(ship_detail)

        # 7. 推进 pipeline，把真实 shipment 放入在途库存
        self._advance_pipeline(shipment_to)

        # 8. 记录历史
        self._record_history(order_qty)

        # 9. reward / metrics
        rewards, holding_costs, backlog_costs = self._compute_rewards(order_qty)
        rewards = self._apply_transport_reward(rewards, transport_costs)
        fill_rates = self._compute_fill_rates()

        self.step_num += 1
        terminated = self.step_num >= self.cfg.episode_len

        info = self._build_info(
            holding_costs, backlog_costs, fill_rates, order_qty,
            shipment_to, transport_costs, routing_solution,
        )

        return self._get_obs(), rewards, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        """Build observation matrix: shape (N, 6 + 3L)."""
        cfg = self.cfg
        max_inv = max(cfg.init_inventory * 3.0, cfg.max_order * cfg.lead_time * 2)
        max_bl = max_inv
        max_dem = cfg.demand_mean * 4.0
        max_pipe = max_dem * cfg.lead_time
        max_order_val = float(cfg.max_order) * 2
        max_layer = max(max(cfg.layers), 1)

        inv_norm = np.clip(self.inventory / max_inv, -1, 1)
        bl_norm = np.clip(self.backlog / max_bl, -1, 1)
        # Original obs used only this demand scalar:
        # dem_norm = np.clip(self.demand / max_dem, -1, 1)
        dem_norm = np.clip(self.demand / max_dem, -1, 1)
        req_norm = np.clip(self.downstream_order_demand / max_dem, -1, 1)
        ext_norm = np.clip(self.external_demand / max_dem, -1, 1)
        layer_norm = 2.0 * (np.array(cfg.layers, dtype=np.float64) / max_layer) - 1.0
        pipe_norm = np.clip(self.pipeline / max_pipe, -1, 1) if max_pipe > 0 else self.pipeline
        dhist_norm = np.clip(self.demand_hist / max_dem, -1, 1) if max_dem > 0 else self.demand_hist
        ohist_norm = np.clip(self.order_hist / max_order_val, -1, 1) if max_order_val > 0 else self.order_hist

        # Original observation:
        # obs = np.concatenate([
        #     inv_norm[:, None], bl_norm[:, None], dem_norm[:, None],
        #     pipe_norm, dhist_norm, ohist_norm,
        # ], axis=1)
        obs = np.concatenate([
            inv_norm[:, None],
            bl_norm[:, None],
            dem_norm[:, None],
            req_norm[:, None],
            ext_norm[:, None],
            layer_norm[:, None],
            pipe_norm,
            dhist_norm,
            ohist_norm,
        ], axis=1)

        return obs.astype(np.float64)
