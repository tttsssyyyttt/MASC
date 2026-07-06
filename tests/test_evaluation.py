"""Test evaluation module: metrics, runner."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from meirp_project.core.config import EnvConfig
from meirp_project.core.env import MEIRPEnv
from meirp_project.evaluation.metrics import compute_metrics
from meirp_project.evaluation.runner import EvalRunner
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.algorithms.iql.agent import IQLAgent

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


def make_cfg():
    return EnvConfig(
        num_nodes=3, edges=[(0,1),(0,2)], layers=[0,1,1],
        H=[1,2,2], B=[5,4,4],
        max_order=10, lead_time=2, episode_len=10,
        demand_mean=5.0, seed=42,
    )


# ===== Metrics Tests =====
print("\n=== Metrics Tests ===")

# Create fake info history
info_history = []
for _ in range(5):
    info_history.append({
        "holding_costs": np.array([10.0, 8.0, 6.0]),
        "backlog_costs": np.array([5.0, 2.0, 0.0]),
        "fill_rates": np.array([0.8, 0.9, 1.0]),
        "order_qty": np.array([5.0, 3.0, 4.0]),
        "inventory": np.array([15.0, 12.0, 10.0]),
        "demand": np.array([6.0, 4.0, 3.0]),
    })

m = compute_metrics(info_history)
check("total_cost computed", "total_cost" in m)
check("total_holding computed", "total_holding" in m)
check("total_backlog computed", "total_backlog" in m)
check("avg_fill_rate computed", "avg_fill_rate" in m)
check("total_cost > 0", m["total_cost"] > 0)
check("avg_fill_rate in [0,1]", 0 <= m["avg_fill_rate"] <= 1)
check("n_steps = 5", m["n_steps"] == 5)

# Check expected values
expected_holding = 5 * (10 + 8 + 6)
check("total_holding correct", abs(m["total_holding"] - expected_holding) < 0.01)
expected_backlog = 5 * (5 + 2 + 0)
check("total_backlog correct", abs(m["total_backlog"] - expected_backlog) < 0.01)

# Empty history
m_empty = compute_metrics([])
check("empty history returns empty", m_empty == {})


# ===== EvalRunner Tests =====
print("\n=== EvalRunner Tests ===")

cfg = make_cfg()
runner = EvalRunner(cfg, n_seeds=3)

# Test eval_baseline
bs = BaseStockPolicy(cfg)
baseline_results = runner.eval_baseline(bs, n_episodes=3)
check("baseline results has total_cost", "total_cost" in baseline_results)
check("baseline results has avg_fill_rate", "avg_fill_rate" in baseline_results)
check("baseline n_episodes = 3", baseline_results.get("n_episodes") == 3)
check("baseline avg_fill_rate in [0,1]", 0 <= baseline_results.get("avg_fill_rate", -1) <= 1)

# Test eval_agent
obs_dim = 6 + 3 * cfg.lead_time
iql = IQLAgent(cfg=cfg, obs_dim=obs_dim, hidden_dim=32, buffer_capacity=10)
agent_results = runner.eval_agent(iql, n_episodes=2, deterministic=True)
check("agent results has total_cost", "total_cost" in agent_results)
check("agent results has avg_fill_rate", "avg_fill_rate" in agent_results)


# ===== Full Episode Metrics Test =====
print("\n=== Full Episode Metrics Test ===")

env = MEIRPEnv(cfg)
obs, _ = env.reset()
info_hist = []

while True:
    actions = bs.get_actions(env)
    obs, rewards, terminated, truncated, info = env.step(actions)
    info_hist.append(info)
    if terminated:
        break

full_metrics = compute_metrics(info_hist)
check("full episode metrics computed", "total_cost" in full_metrics)
check("full episode avg_cost_per_step", "avg_cost_per_step" in full_metrics)
check("full episode per_agent_fill_rate shape", full_metrics["per_agent_fill_rate"].shape == (3,))


# ===== Summary =====
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
if failed > 0:
    sys.exit(1)
else:
    print("All tests passed!")
