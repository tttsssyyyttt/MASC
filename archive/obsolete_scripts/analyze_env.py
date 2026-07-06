"""Analyze environment dynamics to understand high stockout rate."""
import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv

cfg = from_yaml('meirp_project/configs/env_3layers.yaml')
env = MEIRPEnv(cfg, np.random.default_rng(42))

print("=" * 70)
print("  1. 环境参数分析")
print("=" * 70)
print(f"节点数: {cfg.num_nodes}")
print(f"层数: 3 (root=0,1  mid=2,3,4  terminal=5,6,7,8)")
print(f"H (库存成本): {cfg.H}")
print(f"B (缺货成本): {cfg.B}")
bh_ratio = [b / h if h > 0 else float('inf') for b, h in zip(cfg.B, cfg.H)]
print(f"B/H 比值:   {[f'{r:.1f}' for r in bh_ratio]}")
print(f"max_order: {cfg.max_order}")
print(f"lead_time: {cfg.lead_time}")
print(f"demand_mean: {cfg.demand_mean}")

# 上游供应能力
print(f"\n上游关系:")
for i in range(cfg.num_nodes):
    ups = env.upstream[i]
    if len(ups) == 0:
        print(f"  Node {i}: ROOT (外部供应商, max_order={cfg.max_order})")
    else:
        max_supply = cfg.max_order * len(ups)
        print(f"  Node {i}: upstream={ups}, max_supply/step={max_supply}")

# 模拟贪心策略
print("\n" + "=" * 70)
print("  2. 模拟贪心策略 (每步最大订货) — 理论上限")
print("=" * 70)
env1 = MEIRPEnv(cfg, np.random.default_rng(42))
obs, _ = env1.reset()
total_back = np.zeros(cfg.num_nodes)
total_dem = np.zeros(cfg.num_nodes)

for step in range(cfg.episode_len):
    actions = []
    for i in range(env1.n):
        k = len(env1.upstream[i])
        if k == 0:
            actions.append(cfg.max_order)
        else:
            actions.append([cfg.max_order] * k)

    obs, rewards, done, truncated, info = env1.step(actions)
    total_back += env1.backlog
    total_dem += env1.demand

    if step < 3 or step >= cfg.episode_len - 2:
        t = env1.terminals
        print(f"  Step {step:2d}: inv={env1.inventory[t].astype(int)} "
              f"back={env1.backlog[t].astype(int)} "
              f"dem={env1.demand[t].astype(int)} "
              f"order={info['order_qty'][t].astype(int)}")

t = env1.terminals
avg_fill = 1.0 - total_back[t].sum() / (total_dem[t].sum() + 1e-9)
print(f"\n贪心策略 - 终端满足率: {avg_fill:.4f}")
print(f"终端总需求: {total_dem[t].sum():.0f}, 终端总缺货: {total_back[t].sum():.0f}")

# RL 智能订货策略（看库存水平决定订多少）
print("\n" + "=" * 70)
print("  3. 模拟智能策略 (根据库存水平订货)")
print("=" * 70)
env2 = MEIRPEnv(cfg, np.random.default_rng(42))
obs, _ = env2.reset()
total_back2 = np.zeros(cfg.num_nodes)
total_dem2 = np.zeros(cfg.num_nodes)

for step in range(cfg.episode_len):
    actions = []
    for i in range(env2.n):
        k = len(env2.upstream[i])
        # 策略: 订货量 = max(0, 目标库存 - 当前库存 - 在途)
        target = cfg.demand_mean * (cfg.lead_time + 1) * 1.5  # 安全库存系数1.5
        on_hand = env2.inventory[i]
        in_transit = env2.pipeline[i].sum()
        need = max(0, target - on_hand - in_transit + env2.backlog[i])
        order = min(int(need), cfg.max_order)

        if k == 0:
            actions.append(order)
        else:
            per_up = max(1, order // k)
            actions.append([min(per_up, cfg.max_order)] * k)

    obs, rewards, done, truncated, info = env2.step(actions)
    total_back2 += env2.backlog
    total_dem2 += env2.demand

    if step < 3 or step >= cfg.episode_len - 2:
        t = env2.terminals
        print(f"  Step {step:2d}: inv={env2.inventory[t].astype(int)} "
              f"back={env2.backlog[t].astype(int)} "
              f"dem={env2.demand[t].astype(int)} "
              f"order={info['order_qty'][t].astype(int)}")

t = env2.terminals
avg_fill2 = 1.0 - total_back2[t].sum() / (total_dem2[t].sum() + 1e-9)
print(f"\n智能策略 - 终端满足率: {avg_fill2:.4f}")
print(f"终端总需求: {total_dem2[t].sum():.0f}, 终端总缺货: {total_back2[t].sum():.0f}")

# 奖励函数分析
print("\n" + "=" * 70)
print("  4. 奖励函数问题分析")
print("=" * 70)
print("\n终端节点 (H=3, B=3, B/H=1.0):")
print(f"  inv=30, backlog=0  → cost = 3*30 + 3*0  = 90")
print(f"  inv=0,  backlog=30 → cost = 3*0  + 3*30 = 90")
print(f"  → 库存30单位和缺货30单位的惩罚完全相同！")
print(f"  → Agent 没有任何动力多存货来避免缺货！")

print("\n中间层 (H=2, B=4, B/H=2.0):")
print(f"  inv=30, backlog=0  → cost = 60")
print(f"  inv=0,  backlog=30 → cost = 120")
print(f"  → 缺货比库存贵2倍，有一点动力")

print("\n根节点 (H=1, B=5, B/H=5.0):")
print(f"  inv=30, backlog=0  → cost = 30")
print(f"  inv=0,  backlog=30 → cost = 150")
print(f"  → 缺货比库存贵5倍，有强动力")

print("\n" + "=" * 70)
print("  核心问题: 终端节点 B/H = 1.0, RL无法学到避免缺货的策略!")
print("  解决方案: 1) 提高终端B值  2) 添加缺货惩罚项  3) 添加满足率奖励")
print("=" * 70)
