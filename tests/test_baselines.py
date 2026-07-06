"""Test baselines: BaseStock and (s,S) policies."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from meirp_project.core.config import EnvConfig
from meirp_project.core.env import MEIRPEnv
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.baselines.ss_policy import SSPolicy

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


def make_3layer_cfg():
    return EnvConfig(
        num_nodes=7,
        edges=[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        layers=[0, 1, 1, 2, 2, 2, 2],
        H=[1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
        B=[5.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0],
        max_order=20, lead_time=2, episode_len=10,
        demand_mean=8.0, seed=42,
    )


# ===== BaseStock =====
print("\n=== BaseStock Tests ===")

cfg = make_3layer_cfg()
env = MEIRPEnv(cfg)
bs = BaseStockPolicy(cfg)

# Test default S levels
expected_S = cfg.demand_mean * (cfg.lead_time + 1)
check("default S level", all(bs.S == expected_S))

# Test custom S levels
bs_custom = BaseStockPolicy(cfg, base_stock_levels=[30.0] * 7)
check("custom S level", all(bs_custom.S == 30.0))

# Test get_actions (BDQ format: root -> int, non-root -> list[int])
env.reset()
actions = bs.get_actions(env)
check("actions length", len(actions) == 7)

for i in range(7):
    a = actions[i]
    q = sum(a) if isinstance(a, list) else a
    check(f"action q non-negative (node {i})", q >= 0)
    check(f"action q <= max_order (node {i})", q <= cfg.max_order)

# Test run_episode
metrics = bs.run_episode(env)
check("total_reward is finite", np.isfinite(metrics["total_reward"]))
check("avg_fill_rate in [0,1]", 0 <= metrics["avg_fill_rate"] <= 1)
check("total_holding >= 0", metrics["total_holding"] >= 0)
check("total_backlog >= 0", metrics["total_backlog"] >= 0)
check("steps > 0", metrics["avg_reward"] != 0 or metrics["total_holding"] > 0)


# ===== (s,S) Policy =====
print("\n=== (s,S) Policy Tests ===")

ss = SSPolicy(cfg)

# Test default s and S
expected_s = cfg.demand_mean * cfg.lead_time
expected_S = cfg.demand_mean * (cfg.lead_time + 1)
check("default s level", all(ss.s == expected_s))
check("default S level", all(ss.S == expected_S))

# Test custom s and S
ss_custom = SSPolicy(cfg, reorder_points=[15.0]*7, order_up_to_levels=[25.0]*7)
check("custom s level", all(ss_custom.s == 15.0))
check("custom S level", all(ss_custom.S == 25.0))

# Test get_actions (BDQ format)
env.reset()
actions_ss = ss.get_actions(env)
check("ss actions length", len(actions_ss) == 7)

# With high initial inventory, s,S should order 0 (position > s)
for i in range(7):
    a = actions_ss[i]
    q = sum(a) if isinstance(a, list) else a
    check(f"ss no order when position > s (node {i})", q == 0)

# Test run_episode
metrics_ss = ss.run_episode(env)
check("ss total_reward is finite", np.isfinite(metrics_ss["total_reward"]))
check("ss avg_fill_rate in [0,1]", 0 <= metrics_ss["avg_fill_rate"] <= 1)

# ===== Compare policies =====
print("\n=== Policy Comparison ===")

env_bs = MEIRPEnv(make_3layer_cfg())
env_ss = MEIRPEnv(make_3layer_cfg())

m_bs = bs.run_episode(env_bs)
m_ss = ss.run_episode(env_ss)

check("both produce valid rewards",
      np.isfinite(m_bs["total_reward"]) and np.isfinite(m_ss["total_reward"]))

print(f"\n  BaseStock: total_reward={m_bs['total_reward']:.1f}, fill_rate={m_bs['avg_fill_rate']:.3f}")
print(f"  (s,S):     total_reward={m_ss['total_reward']:.1f}, fill_rate={m_ss['avg_fill_rate']:.3f}")


# ===== Summary =====
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
if failed > 0:
    sys.exit(1)
else:
    print("All tests passed!")
