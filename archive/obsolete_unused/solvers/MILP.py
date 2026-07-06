#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三级供应链订单分配与车辆路径问题（MILP）- 函数封装版
支持任意节点数量（需保证坐标、层级划分合理）
输入：数据字典，包含所有参数
输出：求解状态、目标值、订单分配、库存积压、路径等
"""

from ortools.sat.python import cp_model
import math
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np  # 用于生成随机坐标
import time


# ==================== 数据类定义 ====================
@dataclass
class SupplyChainData:
    """供应链问题输入数据"""
    # 节点信息
    top_nodes: List[int]  # 顶层节点原始编号列表
    mid_nodes: List[int]  # 中层节点原始编号列表
    bot_nodes: List[int]  # 底层节点原始编号列表
    coords: Dict[int, Tuple[float, float]]  # 每个原始节点的坐标 (x,y)

    # 各层参数（顺序与节点列表一致）
    I0_top: List[float]  # 顶层期初库存
    B0_top: List[float]  # 顶层期初积压
    h_top: List[float]  # 顶层库存持有成本
    p_top: List[float]  # 顶层积压惩罚
    incoming_top: List[float]  # 顶层本周期在途到货
    total_order_top: List[float]

    I0_mid: List[float]
    B0_mid: List[float]
    h_mid: List[float]
    p_mid: List[float]
    incoming_mid: List[float]
    total_order_mid: List[float]  # 中层向顶层下的总订单量

    I0_bot: List[float]
    B0_bot: List[float]
    h_bot: List[float]
    p_bot: List[float]
    incoming_bot: List[float]
    total_order_bot: List[float]  # 底层向中层下的总订单量
    final_demand_bot: List[float]  # 底层最终需求

    # 车辆容量
    C1: float  # 顶层车辆容量（所有顶层车辆相同）
    C2: float  # 中层车辆容量（所有中层车辆相同）

    # 求解参数
    time_limit_sec: float = 60.0  # 求解时间限制
    scale: int = 100  # 内部缩放因子


# ==================== 核心求解函数 ====================
def solve_supply_chain(data: SupplyChainData) -> Dict:
    """
    求解三级供应链订单分配与车辆路径问题
    返回字典包含：
        status: str (OPTIMAL/FEASIBLE/INFEASIBLE/UNKNOWN)
        objective: float (原始目标值)
        assignment_top: Dict[(int,int), float] 顶层→中层的分配量
        assignment_mid: Dict[(int,int), float] 中层→底层的分配量
        inventory_top: List[float]
        backlog_top: List[float]
        inventory_mid: List[float]
        backlog_mid: List[float]
        inventory_bot: List[float]
        backlog_bot: List[float]
        routes_top: List[List[int]]  每个顶层车辆的路径（原始节点号）
        routes_mid: List[List[int]]  每个中层车辆的路径
        solve_time: float
    """
    # 提取数据
    top_nodes = data.top_nodes
    mid_nodes = data.mid_nodes
    bot_nodes = data.bot_nodes
    coords = data.coords
    all_nodes = top_nodes + mid_nodes + bot_nodes
    N = len(all_nodes)

    # 建立映射：原始编号 -> 内部索引 (0..N-1)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    S = [node_to_idx[n] for n in top_nodes]
    J = [node_to_idx[n] for n in mid_nodes]
    K = [node_to_idx[n] for n in bot_nodes]

    # 计算距离矩阵（浮点数）
    dist = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            xi, yi = coords[idx_to_node[i]]
            xj, yj = coords[idx_to_node[j]]
            dist[i][j] = math.hypot(xi - xj, yi - yj)

    # 缩放参数
    SCALE = data.scale

    def scale(x: float) -> int:
        return int(round(x * SCALE))

    def unscale(x: int) -> float:
        return x / SCALE

    dist_int = [[scale(dist[i][j]) for j in range(N)] for i in range(N)]
    C1_int = scale(data.C1)
    C2_int = scale(data.C2)

    # 缩放各参数
    I0_top_int = [scale(v) for v in data.I0_top]
    B0_top_int = [scale(v) for v in data.B0_top]
    h_top_int = [scale(v) for v in data.h_top]
    p_top_int = [scale(v) for v in data.p_top]
    incoming_top_int = [scale(v) for v in data.incoming_top]

    I0_mid_int = [scale(v) for v in data.I0_mid]
    B0_mid_int = [scale(v) for v in data.B0_mid]
    h_mid_int = [scale(v) for v in data.h_mid]
    p_mid_int = [scale(v) for v in data.p_mid]
    incoming_mid_int = [scale(v) for v in data.incoming_mid]
    total_order_mid_int = [scale(v) for v in data.total_order_mid]

    I0_bot_int = [scale(v) for v in data.I0_bot]
    B0_bot_int = [scale(v) for v in data.B0_bot]
    h_bot_int = [scale(v) for v in data.h_bot]
    p_bot_int = [scale(v) for v in data.p_bot]
    incoming_bot_int = [scale(v) for v in data.incoming_bot]
    total_order_bot_int = [scale(v) for v in data.total_order_bot]
    final_demand_bot_int = [scale(v) for v in data.final_demand_bot]

    # 大M（变量上界）
    M_int = scale(1000)  # 足够大

    # 创建模型
    model = cp_model.CpModel()

    # 决策变量
    O = {}
    for s in S:
        for j in J:
            O[(s, j)] = model.NewIntVar(0, C1_int, f'O_{idx_to_node[s]}_{idx_to_node[j]}')
    P = {}
    for j in J:
        for k in K:
            P[(j, k)] = model.NewIntVar(0, C2_int, f'P_{idx_to_node[j]}_{idx_to_node[k]}')

    I_top = [model.NewIntVar(0, M_int, f'I_top_{idx_to_node[s]}') for s in S]
    B_top = [model.NewIntVar(0, M_int, f'B_top_{idx_to_node[s]}') for s in S]
    I_mid = [model.NewIntVar(0, M_int, f'I_mid_{idx_to_node[j]}') for j in J]
    B_mid = [model.NewIntVar(0, M_int, f'B_mid_{idx_to_node[j]}') for j in J]
    I_bot = [model.NewIntVar(0, M_int, f'I_bot_{idx_to_node[k]}') for k in K]
    B_bot = [model.NewIntVar(0, M_int, f'B_bot_{idx_to_node[k]}') for k in K]

    # 路径变量
    x = {}
    for s in S:
        nodes_s = [s] + J
        for u in nodes_s:
            for v in nodes_s:
                if u != v:
                    x[(s, u, v)] = model.NewBoolVar(f'x_{idx_to_node[s]}_{idx_to_node[u]}_{idx_to_node[v]}')
    y = {}
    for j in J:
        nodes_j = [j] + K
        for u in nodes_j:
            for v in nodes_j:
                if u != v:
                    y[(j, u, v)] = model.NewBoolVar(f'y_{idx_to_node[j]}_{idx_to_node[u]}_{idx_to_node[v]}')

    # ---------- 约束 ----------
    # 1. 总订单量约束
    for idx_j, j in enumerate(J):
        model.Add(sum(O[(s, j)] for s in S) == total_order_mid_int[idx_j])
    for idx_k, k in enumerate(K):
        model.Add(sum(P[(j, k)] for j in J) == total_order_bot_int[idx_k])

    # 2. 库存平衡
    for idx_s, s in enumerate(S):
        demand = sum(O[(s, j)] for j in J)
        model.Add(I_top[idx_s] - B_top[idx_s] ==
                  I0_top_int[idx_s] - B0_top_int[idx_s] + incoming_top_int[idx_s] - demand)
    for idx_j, j in enumerate(J):
        demand = sum(P[(j, k)] for k in K)
        model.Add(I_mid[idx_j] - B_mid[idx_j] ==
                  I0_mid_int[idx_j] - B0_mid_int[idx_j] + incoming_mid_int[idx_j] - demand)
    for idx_k, k in enumerate(K):
        model.Add(I_bot[idx_k] - B_bot[idx_k] ==
                  I0_bot_int[idx_k] - B0_bot_int[idx_k] + incoming_bot_int[idx_k] - final_demand_bot_int[idx_k])

    # 3. 车辆容量（安全边界，但总订单量已约束，可省略）
    for s in S:
        model.Add(sum(O[(s, j)] for j in J) <= C1_int)
    for j in J:
        model.Add(sum(P[(j, k)] for k in K) <= C2_int)

    # 4. 车辆路径（哈密顿回路）
    for s in S:
        nodes_s = [s] + J
        arcs = [(u, v, x[(s, u, v)]) for u in nodes_s for v in nodes_s if u != v]
        model.AddCircuit(arcs)
    for j in J:
        nodes_j = [j] + K
        arcs = [(u, v, y[(j, u, v)]) for u in nodes_j for v in nodes_j if u != v]
        model.AddCircuit(arcs)

    # ---------- 目标函数 ----------
    distance_cost = 0
    for s in S:
        nodes_s = [s] + J
        for u in nodes_s:
            for v in nodes_s:
                if u != v:
                    distance_cost += dist_int[u][v] * x[(s, u, v)]
    for j in J:
        nodes_j = [j] + K
        for u in nodes_j:
            for v in nodes_j:
                if u != v:
                    distance_cost += dist_int[u][v] * y[(j, u, v)]

    inventory_cost = 0
    for idx_s, s in enumerate(S):
        inventory_cost += h_top_int[idx_s] * I_top[idx_s] + p_top_int[idx_s] * B_top[idx_s]
    for idx_j, j in enumerate(J):
        inventory_cost += h_mid_int[idx_j] * I_mid[idx_j] + p_mid_int[idx_j] * B_mid[idx_j]
    for idx_k, k in enumerate(K):
        inventory_cost += h_bot_int[idx_k] * I_bot[idx_k] + p_bot_int[idx_k] * B_bot[idx_k]

    model.Minimize(distance_cost + inventory_cost)

    # 求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = data.time_limit_sec
    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time

    # 解析结果
    result = {
        'status': solver.StatusName(status),
        'objective': None,
        'assignment_top': {},
        'assignment_mid': {},
        'inventory_top': [],
        'backlog_top': [],
        'inventory_mid': [],
        'backlog_mid': [],
        'inventory_bot': [],
        'backlog_bot': [],
        'routes_top': [],
        'routes_mid': [],
        'solve_time': solve_time
    }

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        result['objective'] = unscale(solver.ObjectiveValue())

        # 订单分配
        for s in S:
            for j in J:
                val = solver.Value(O[(s, j)])
                if val > 0:
                    result['assignment_top'][(idx_to_node[s], idx_to_node[j])] = unscale(val)
        for j in J:
            for k in K:
                val = solver.Value(P[(j, k)])
                if val > 0:
                    result['assignment_mid'][(idx_to_node[j], idx_to_node[k])] = unscale(val)

        # 库存积压
        for idx_s, s in enumerate(S):
            result['inventory_top'].append(unscale(solver.Value(I_top[idx_s])))
            result['backlog_top'].append(unscale(solver.Value(B_top[idx_s])))
        for idx_j, j in enumerate(J):
            result['inventory_mid'].append(unscale(solver.Value(I_mid[idx_j])))
            result['backlog_mid'].append(unscale(solver.Value(B_mid[idx_j])))
        for idx_k, k in enumerate(K):
            result['inventory_bot'].append(unscale(solver.Value(I_bot[idx_k])))
            result['backlog_bot'].append(unscale(solver.Value(B_bot[idx_k])))

        # 路径重建（顶层车辆）
        for s in S:
            nodes_s = [s] + J
            next_node = {}
            for u in nodes_s:
                for v in nodes_s:
                    if u != v and solver.Value(x[(s, u, v)]) == 1:
                        next_node[u] = v
            if not next_node:
                continue
            path = []
            cur = s
            while True:
                path.append(idx_to_node[cur])
                cur = next_node[cur]
                if cur == s:
                    break
            path.append(idx_to_node[s])
            result['routes_top'].append(path)

        # 路径重建（中层车辆）
        for j in J:
            nodes_j = [j] + K
            next_node = {}
            for u in nodes_j:
                for v in nodes_j:
                    if u != v and solver.Value(y[(j, u, v)]) == 1:
                        next_node[u] = v
            if not next_node:
                continue
            path = []
            cur = j
            while True:
                path.append(idx_to_node[cur])
                cur = next_node[cur]
                if cur == j:
                    break
            path.append(idx_to_node[j])
            result['routes_mid'].append(path)
        # # 可视化（可选，需安装matplotlib）
        # try:
        #     import matplotlib.pyplot as plt
        #
        #     def plot_routes():
        #         node_coords = {node: coords[node] for node in range(1, N + 1)}
        #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        #         # 顶层车辆
        #         ax1.set_title("顶层车辆路径")
        #         for node, (x_, y_) in node_coords.items():
        #             ax1.scatter(x_, y_, s=100, c='black')
        #             ax1.annotate(str(node), (x_, y_), textcoords="offset points", xytext=(5, 5))
        #         colors = ['red', 'blue', 'green']
        #         for idx_s, s in enumerate(S):
        #             nodes_s = [s] + J
        #             next_node = {}
        #             for u in nodes_s:
        #                 for v in nodes_s:
        #                     if u != v and solver.Value(x[(s, u, v)]) == 1:
        #                         next_node[u] = v
        #             if not next_node: continue
        #             path = []
        #             cur = s
        #             while True:
        #                 path.append(cur)
        #                 cur = next_node[cur]
        #                 if cur == s: break
        #             path.append(s)
        #             path_orig = [idx_to_node[n] for n in path]
        #             xs = [node_coords[n][0] for n in path_orig]
        #             ys = [node_coords[n][1] for n in path_orig]
        #             ax1.plot(xs, ys, marker='o', color=colors[idx_s % len(colors)], linewidth=2, label=f'顶层车辆 {idx_to_node[s]}')
        #         ax1.legend()
        #         ax1.grid(True)
        #         # 中层车辆
        #         ax2.set_title("中层车辆路径")
        #         for node, (x_, y_) in node_coords.items():
        #             ax2.scatter(x_, y_, s=100, c='black')
        #             ax2.annotate(str(node), (x_, y_), textcoords="offset points", xytext=(5, 5))
        #         colors_mid = ['orange', 'purple']
        #         for idx_j, j in enumerate(J):
        #             nodes_j = [j] + K
        #             next_node = {}
        #             for u in nodes_j:
        #                 for v in nodes_j:
        #                     if u != v and solver.Value(y[(j, u, v)]) == 1:
        #                         next_node[u] = v
        #             if not next_node: continue
        #             path = []
        #             cur = j
        #             while True:
        #                 path.append(cur)
        #                 cur = next_node[cur]
        #                 if cur == j: break
        #             path.append(j)
        #             path_orig = [idx_to_node[n] for n in path]
        #             xs = [node_coords[n][0] for n in path_orig]
        #             ys = [node_coords[n][1] for n in path_orig]
        #             ax2.plot(xs, ys, marker='o', color=colors_mid[idx_j % len(colors_mid)], linewidth=2, label=f'中层车辆 {idx_to_node[j]}')
        #         ax2.legend()
        #         ax2.grid(True)
        #         plt.tight_layout()
        #         plt.show()
        #
        #     plot_routes()
        # except ImportError:
        #     print("matplotlib未安装，跳过绘图。")
    return result


@dataclass
class SupplyChainConfig:
    """静态配置"""
    top_nodes: List[int]  # 顶层原始编号
    mid_nodes: List[int]  # 中层原始编号
    bot_nodes: List[int]  # 底层原始编号
    coords: Dict[int, Tuple[float, float]]  # 每个节点的坐标
    L: int = 2  # 提前期（周期数）
    C1: float = 30.0  # 顶层车辆容量（也作为顶层外部订单上限）
    C2: float = 30.0  # 中层车辆容量
    h_top: List[float] = field(default_factory=list)  # 顶层库存成本
    p_top: List[float] = field(default_factory=list)  # 顶层积压成本
    h_mid: List[float] = field(default_factory=list)
    p_mid: List[float] = field(default_factory=list)
    h_bot: List[float] = field(default_factory=list)
    p_bot: List[float] = field(default_factory=list)
    scale: int = 100  # 缩放因子（浮点转整数）
    time_limit_sec: float = 60.0  # 求解时间限制


@dataclass
class SystemState:
    """当前周期开始前的系统状态"""
    I_top: List[float]  # 库存
    B_top: List[float]  # 积压
    I_mid: List[float]
    B_mid: List[float]
    I_bot: List[float]
    B_bot: List[float]
    on_order_top: List[List[float]]  # 顶层在途队列，长度 = L
    on_order_mid: List[List[float]]  # 中层在途队列
    on_order_bot: List[List[float]]  # 底层在途队列
    current_period: int = 0


class RollingHorizonSolver:
    def __init__(self, config: SupplyChainConfig):
        self.config = config
        # 建立节点内部索引（0-based）
        all_nodes = config.top_nodes + config.mid_nodes + config.bot_nodes
        self.node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        self.idx_to_node = {i: node for node, i in self.node_to_idx.items()}
        self.S = [self.node_to_idx[n] for n in config.top_nodes]
        self.J = [self.node_to_idx[n] for n in config.mid_nodes]
        self.K = [self.node_to_idx[n] for n in config.bot_nodes]
        self.N = len(all_nodes)
        # 距离矩阵（浮点数）
        self.dist = [[0.0] * self.N for _ in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                xi, yi = config.coords[i + 1]
                xj, yj = config.coords[j + 1]
                self.dist[i][j] = math.hypot(xi - xj, yi - yj)

    def _scale(self, x: float) -> int:
        return int(round(x * self.config.scale))

    def _unscale(self, x: int) -> float:
        return x / self.config.scale

    def solve_period(
            self,
            state: SystemState,
            forecast_demand: List[List[float]],  # 预测需求 [t][底层索引]
            horizon: int,
            rl_action: Optional[Tuple[List[float], List[float], List[float]]] = None
    ) -> Dict:
        """
        求解滚动窗口 MILP
        :param state: 当前状态
        :param forecast_demand: 未来 horizon 个周期的底层需求（浮点数）
        :param horizon: 规划周期数
        :param rl_action: 如果提供，则为 (top_orders, mid_orders, bot_orders)
                          top_orders: 长度 len(S) 的列表，本周期顶层外部订单量
                          mid_orders: 长度 len(J) 的列表，本周期中层总订单量（向顶层）
                          bot_orders: 长度 len(K) 的列表，本周期底层总订单量（向中层）
                          若为 None，则本周期所有总订单量由模型自由优化（基线模式）
        :return: 包含当前周期决策的字典
        """
        T = horizon
        L = self.config.L
        SCALE = self.config.scale

        model = cp_model.CpModel()

        # ---------- 创建变量 ----------
        # 订单分配：O[(s, j, t)], P[(j, k, t)]
        O = {}
        P = {}
        # 库存与积压
        I_top = [[None] * T for _ in self.S]
        B_top = [[None] * T for _ in self.S]
        I_mid = [[None] * T for _ in self.J]
        B_mid = [[None] * T for _ in self.J]
        I_bot = [[None] * T for _ in self.K]
        B_bot = [[None] * T for _ in self.K]
        # 动作（发出的总订单量）
        action_top = [[None] * T for _ in self.S]  # 顶层向外部下订单
        action_mid = [[None] * T for _ in self.J]  # 中层向顶层下订单
        action_bot = [[None] * T for _ in self.K]  # 底层向中层下订单
        # 到货量（由动作延迟L周期得到）
        incoming_top = [[None] * T for _ in self.S]
        incoming_mid = [[None] * T for _ in self.J]
        incoming_bot = [[None] * T for _ in self.K]
        # 路径变量
        x = {}  # x[(s, u, v, t)]
        y = {}  # y[(j, u, v, t)]


        C1_int = self.config.C1
        C2_int = self.config.C2
        M_int = 1000
        # 在决策变量区域添加偏差变量（在创建 I_top 之后）
        dev_top = [[model.NewIntVar(0, M_int, f'dev_top_{t}_{idx_s}') for t in range(T)] for idx_s in range(len(self.S))]
        dev_mid = [[model.NewIntVar(0, M_int, f'dev_mid_{t}_{idx_j}') for t in range(T)] for idx_j in range(len(self.J))]
        dev_bot = [[model.NewIntVar(0, M_int, f'dev_bot_{t}_{idx_k}') for t in range(T)] for idx_k in range(len(self.K))]

        for t in range(T):
            # 订单分配
            for s in self.S:
                for j in self.J:
                    O[(s, j, t)] = model.NewIntVar(0, C1_int, f'O_{t}_{s}_{j}')
            for j in self.J:
                for k in self.K:
                    P[(j, k, t)] = model.NewIntVar(0, C2_int, f'P_{t}_{j}_{k}')

            # 库存积压
            for idx_s in range(len(self.S)):
                I_top[idx_s][t] = model.NewIntVar(0, M_int, f'I_top_{t}_{idx_s}')
                B_top[idx_s][t] = model.NewIntVar(0, M_int, f'B_top_{t}_{idx_s}')
            for idx_j in range(len(self.J)):
                I_mid[idx_j][t] = model.NewIntVar(0, M_int, f'I_mid_{t}_{idx_j}')
                B_mid[idx_j][t] = model.NewIntVar(0, M_int, f'B_mid_{t}_{idx_j}')
            for idx_k in range(len(self.K)):
                I_bot[idx_k][t] = model.NewIntVar(0, M_int, f'I_bot_{t}_{idx_k}')
                B_bot[idx_k][t] = model.NewIntVar(0, M_int, f'B_bot_{t}_{idx_k}')

            # 动作（总订单量）
            for idx_s in range(len(self.S)):
                action_top[idx_s][t] = model.NewIntVar(0, C1_int, f'act_top_{t}_{idx_s}')
            for idx_j in range(len(self.J)):
                action_mid[idx_j][t] = model.NewIntVar(0, C1_int, f'act_mid_{t}_{idx_j}')
            for idx_k in range(len(self.K)):
                action_bot[idx_k][t] = model.NewIntVar(0, C2_int, f'act_bot_{t}_{idx_k}')

            # 到货量
            for idx_s in range(len(self.S)):
                incoming_top[idx_s][t] = model.NewIntVar(0, C1_int, f'inc_top_{t}_{idx_s}')
            for idx_j in range(len(self.J)):
                incoming_mid[idx_j][t] = model.NewIntVar(0, C1_int, f'inc_mid_{t}_{idx_j}')
            for idx_k in range(len(self.K)):
                incoming_bot[idx_k][t] = model.NewIntVar(0, C2_int, f'inc_bot_{t}_{idx_k}')

            # 路径变量
            for s in self.S:
                nodes_s = [s] + self.J
                for u in nodes_s:
                    for v in nodes_s:
                        if u != v:
                            x[(s, u, v, t)] = model.NewBoolVar(f'x_{t}_{s}_{u}_{v}')
            for j in self.J:
                nodes_j = [j] + self.K
                for u in nodes_j:
                    for v in nodes_j:
                        if u != v:
                            y[(j, u, v, t)] = model.NewBoolVar(f'y_{t}_{j}_{u}_{v}')

        # ---------- 约束 ----------
        # 1. 动作与订单分配的关系
        for idx_j, j in enumerate(self.J):
            for t in range(T):
                model.Add(action_mid[idx_j][t] == sum(O[(s, j, t)] for s in self.S))
        for idx_k, k in enumerate(self.K):
            for t in range(T):
                model.Add(action_bot[idx_k][t] == sum(P[(j, k, t)] for j in self.J))
        # 注意：action_top 没有分配关系，它是直接向外部下的订单

        # 2. 到货与动作的延迟关系（提前期 L）
        # 顶层到货
        for idx_s in range(len(self.S)):
            for t in range(T):
                if t < L:
                    model.Add(incoming_top[idx_s][t] == state.on_order_top[idx_s][t])
                else:
                    model.Add(incoming_top[idx_s][t] == action_top[idx_s][t - L])
        # 中层到货
        for idx_j in range(len(self.J)):
            for t in range(T):
                if t < L:
                    model.Add(incoming_mid[idx_j][t] == state.on_order_mid[idx_j][t])
                else:
                    model.Add(incoming_mid[idx_j][t] == action_mid[idx_j][t - L])
        # 底层到货
        for idx_k in range(len(self.K)):
            for t in range(T):
                if t < L:
                    model.Add(incoming_bot[idx_k][t] == state.on_order_bot[idx_k][t])
                else:
                    model.Add(incoming_bot[idx_k][t] == action_bot[idx_k][t - L])

        # 3. 库存平衡
        # 顶层
        for idx_s in range(len(self.S)):
            for t in range(T):
                demand = sum(O[(self.S[idx_s], j, t)] for j in self.J)  # 向中层发货量
                if t == 0:
                    I0_int = state.I_top[idx_s]
                    B0_int = state.B_top[idx_s]
                    model.Add(I_top[idx_s][t] - B_top[idx_s][t] ==
                              I0_int - B0_int + incoming_top[idx_s][t] - demand)
                else:
                    model.Add(I_top[idx_s][t] - B_top[idx_s][t] ==
                              I_top[idx_s][t - 1] - B_top[idx_s][t - 1] + incoming_top[idx_s][t] - demand)
        # 中层
        for idx_j in range(len(self.J)):
            for t in range(T):
                demand = sum(P[(self.J[idx_j], k, t)] for k in self.K)
                if t == 0:
                    I0_int = state.I_mid[idx_j]
                    B0_int = state.B_mid[idx_j]
                    model.Add(I_mid[idx_j][t] - B_mid[idx_j][t] ==
                              I0_int - B0_int + incoming_mid[idx_j][t] - demand)
                else:
                    model.Add(I_mid[idx_j][t] - B_mid[idx_j][t] ==
                              I_mid[idx_j][t - 1] - B_mid[idx_j][t - 1] + incoming_mid[idx_j][t] - demand)
        # 底层
        for idx_k in range(len(self.K)):
            for t in range(T):
                demand_int = forecast_demand[t][idx_k]
                if t == 0:
                    I0_int = state.I_bot[idx_k]
                    B0_int = state.B_bot[idx_k]
                    model.Add(I_bot[idx_k][t] - B_bot[idx_k][t] ==
                              I0_int - B0_int + incoming_bot[idx_k][t] - demand_int)
                else:
                    model.Add(I_bot[idx_k][t] - B_bot[idx_k][t] ==
                              I_bot[idx_k][t - 1] - B_bot[idx_k][t - 1] + incoming_bot[idx_k][t] - demand_int)
        safety_int = 20
        # 在约束区域添加绝对值约束（放在库存平衡约束之后）
        for idx_s in range(len(self.S)):
            for t in range(T):
                model.Add(dev_top[idx_s][t] >= I_top[idx_s][t] - safety_int)
                model.Add(dev_top[idx_s][t] >= safety_int - I_top[idx_s][t])

        for idx_j in range(len(self.J)):
            for t in range(T):
                model.Add(dev_mid[idx_j][t] >= I_mid[idx_j][t] - safety_int)
                model.Add(dev_mid[idx_j][t] >= safety_int - I_mid[idx_j][t])

        for idx_k in range(len(self.K)):
            for t in range(T):
                model.Add(dev_bot[idx_k][t] >= I_bot[idx_k][t] - safety_int)
                model.Add(dev_bot[idx_k][t] >= safety_int - I_bot[idx_k][t])

        # 4. 车辆容量（每周期）
        for s in self.S:
            for t in range(T):
                model.Add(sum(O[(s, j, t)] for j in self.J) <= C1_int)
        for j in self.J:
            for t in range(T):
                model.Add(sum(P[(j, k, t)] for k in self.K) <= C2_int)

        # 5. 车辆路径（每周期哈密顿回路）
        for s in self.S:
            for t in range(T):
                nodes_s = [s] + self.J
                arcs = [(u, v, x[(s, u, v, t)]) for u in nodes_s for v in nodes_s if u != v]
                model.AddCircuit(arcs)
        for j in self.J:
            for t in range(T):
                nodes_j = [j] + self.K
                arcs = [(u, v, y[(j, u, v, t)]) for u in nodes_j for v in nodes_j if u != v]
                model.AddCircuit(arcs)

        # ---------- RL 动作注入（固定 t=0 的总订单量）----------
        if rl_action is not None:
            rl_top, rl_mid, rl_bot = rl_action[:len(self.S)], rl_action[len(self.S):len(self.S) + len(self.J)], rl_action[len(self.S) + len(self.J):]
            assert len(rl_top) == len(self.S), "RL top orders length mismatch"
            assert len(rl_mid) == len(self.J), "RL mid orders length mismatch"
            assert len(rl_bot) == len(self.K), "RL bot orders length mismatch"
            for idx_s in range(len(self.S)):
                val_int = rl_top[idx_s]
                model.Add(action_top[idx_s][0] == val_int)
            for idx_j in range(len(self.J)):
                val_int = rl_mid[idx_j]
                model.Add(action_mid[idx_j][0] == val_int)
            for idx_k in range(len(self.K)):
                val_int = rl_bot[idx_k]
                model.Add(action_bot[idx_k][0] == val_int)
        # 否则（基线模式）不做额外约束，让变量自由

        # ---------- 目标函数 ----------
        distance_cost = 0
        for s in self.S:
            for t in range(T):
                nodes_s = [s] + self.J
                for u in nodes_s:
                    for v in nodes_s:
                        if u != v:
                            distance_cost += self._scale(self.dist[u][v]) * x[(s, u, v, t)]
        for j in self.J:
            for t in range(T):
                nodes_j = [j] + self.K
                for u in nodes_j:
                    for v in nodes_j:
                        if u != v:
                            distance_cost += self._scale(self.dist[u][v]) * y[(j, u, v, t)]

        inventory_cost = 0
        for idx_s in range(len(self.S)):
            for t in range(T):
                h = self.config.h_top[idx_s]
                p = self.config.p_top[idx_s]
                # inventory_cost += h * I_top[idx_s][t] + p * B_top[idx_s][t]
                inventory_cost += h * dev_top[idx_s][t] + p * B_top[idx_s][t]  # 使用 dev 替代 I
        for idx_j in range(len(self.J)):
            for t in range(T):
                h = self.config.h_mid[idx_j]
                p = self.config.p_mid[idx_j]
                # inventory_cost += h * I_mid[idx_j][t] + p * B_mid[idx_j][t]
                inventory_cost += h * dev_mid[idx_j][t] + p * B_mid[idx_j][t]
        for idx_k in range(len(self.K)):
            for t in range(T):
                h = self.config.h_bot[idx_k]
                p = self.config.p_bot[idx_k]
                # inventory_cost += h * I_bot[idx_k][t] + p * B_bot[idx_k][t]
                inventory_cost += h * dev_bot[idx_k][t] + p * B_bot[idx_k][t]

        model.Minimize(distance_cost + inventory_cost)

        # 求解
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.time_limit_sec
        status = solver.Solve(model)

        # 提取结果
        result = {
            'status': solver.StatusName(status),
            'objective': None,
            'O': {},  # 当前周期订单分配（顶层→中层）
            'P': {},  # 当前周期订单分配（中层→底层）
            'action_top': None,  # 当前周期顶层外部订单量
            'action_mid': None,  # 当前周期中层总订单量
            'action_bot': None,  # 当前周期底层总订单量
            'routes_top': [],
            'routes_mid': [],
        }

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            result['objective'] = self._unscale(solver.ObjectiveValue())
            # 当前周期（t=0）的分配
            for s in self.S:
                for j in self.J:
                    val = solver.Value(O[(s, j, 0)])
                    if val > 0:
                        result['O'][(self.idx_to_node[s], self.idx_to_node[j])] = val
            for j in self.J:
                for k in self.K:
                    val = solver.Value(P[(j, k, 0)])
                    if val > 0:
                        result['P'][(self.idx_to_node[j], self.idx_to_node[k])] = val
            # 总订单量（t=0）
            top_orders = [solver.Value(action_top[idx_s][0]) for idx_s in range(len(self.S))]
            mid_orders = [solver.Value(action_mid[idx_j][0]) for idx_j in range(len(self.J))]
            bot_orders = [solver.Value(action_bot[idx_k][0]) for idx_k in range(len(self.K))]
            result['action_top'] = top_orders
            result['action_mid'] = mid_orders
            result['action_bot'] = bot_orders

            # 路径重建（t=0）
            for s in self.S:
                nodes_s = [s] + self.J
                next_node = {}
                for u in nodes_s:
                    for v in nodes_s:
                        if u != v and solver.Value(x[(s, u, v, 0)]) == 1:
                            next_node[u] = v
                if next_node:
                    path = []
                    cur = s
                    while True:
                        path.append(self.idx_to_node[cur])
                        cur = next_node[cur]
                        if cur == s:
                            break
                    path.append(self.idx_to_node[s])
                    result['routes_top'].append(path)
            for j in self.J:
                nodes_j = [j] + self.K
                next_node = {}
                for u in nodes_j:
                    for v in nodes_j:
                        if u != v and solver.Value(y[(j, u, v, 0)]) == 1:
                            next_node[u] = v
                if next_node:
                    path = []
                    cur = j
                    while True:
                        path.append(self.idx_to_node[cur])
                        cur = next_node[cur]
                        if cur == j:
                            break
                    path.append(self.idx_to_node[j])
                    result['routes_mid'].append(path)
        # 在 solve_period 函数内，构建 result 之前，获取所有节点
        all_nodes = len(self.S) + len(self.J) + len(self.K)
        order_matrix = [[0.0] * (all_nodes + 1) for _ in range(all_nodes + 1)]

        # 填充顶层→中层订单 to -> from
        for (s, j), val in result['O'].items():
            order_matrix[self.node_to_idx[s]][self.node_to_idx[j]] = val

        # 填充中层→底层订单
        for (j, k), val in result['P'].items():
            order_matrix[self.node_to_idx[j]][self.node_to_idx[k]] = val

        # 将矩阵加入 result
        result['order_matrix'] = order_matrix
        # # 在求解成功后，打印所有关键变量（每个周期）
        # if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        #     print("\n======= 详细变量值（缩放后还原为原始值） =======")
        #     for t in range(1):
        #         print(f"\n--- 周期 {t} ---")
        #         # 顶层
        #         print("顶层节点：")
        #         for idx_s, s in enumerate(self.S):
        #             node = self.idx_to_node[s]
        #             I_val = solver.Value(I_top[idx_s][t])
        #             B_val = solver.Value(B_top[idx_s][t])
        #             inc_val = solver.Value(incoming_top[idx_s][t])
        #             act_val = solver.Value(action_top[idx_s][t]) if t < T else 0
        #             print(f"  {node}: I={I_val:.2f}, B={B_val:.2f}, incoming={inc_val:.2f}, action_top={act_val:.2f}")
        #         # 中层
        #         print("中层节点：")
        #         for idx_j, j in enumerate(self.J):
        #             node = self.idx_to_node[j]
        #             I_val = solver.Value(I_mid[idx_j][t])
        #             B_val = solver.Value(B_mid[idx_j][t])
        #             inc_val = solver.Value(incoming_mid[idx_j][t])
        #             act_val = solver.Value(action_mid[idx_j][t])
        #             print(f"  {node}: I={I_val:.2f}, B={B_val:.2f}, incoming={inc_val:.2f}, action_mid={act_val:.2f}")
        #         # 底层
        #         print("底层节点：")
        #         for idx_k, k in enumerate(self.K):
        #             node = self.idx_to_node[k]
        #             I_val = solver.Value(I_bot[idx_k][t])
        #             B_val = solver.Value(B_bot[idx_k][t])
        #             inc_val = solver.Value(incoming_bot[idx_k][t])
        #             act_val = solver.Value(action_bot[idx_k][t])
        #             print(f"  {node}: I={I_val:.2f}, B={B_val:.2f}, incoming={inc_val:.2f}, action_bot={act_val:.2f}")
        return result
