"""Run core module tests without pytest dependency."""

import sys
import os
# Add the parent of meirp_project to path so we can import as package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from meirp_project.core.config import EnvConfig
from meirp_project.core.network import build_network
from meirp_project.core.demand import DemandGenerator
from meirp_project.core.allocation import allocate
from meirp_project.core.reward import compute_reward
from meirp_project.core.env import MEIRPEnv

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


def make_2layer_cfg():
    return EnvConfig(
        num_nodes=3, edges=[(0, 1), (0, 2)], layers=[0, 1, 1],
        H=[1.0, 2.0, 2.0], B=[5.0, 4.0, 4.0],
        max_order=20, lead_time=2, episode_len=30, demand_mean=8.0, seed=42,
    )

def make_3layer_cfg():
    return EnvConfig(
        num_nodes=7,
        edges=[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        layers=[0, 1, 1, 2, 2, 2, 2],
        H=[1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
        B=[5.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0],
        max_order=20, lead_time=2, episode_len=30, demand_mean=8.0, seed=42,
    )


# ===== Config =====
print("\n=== Config Tests ===")

cfg = make_3layer_cfg()
check("valid config num_nodes", cfg.num_nodes == 7)
check("valid config edges", len(cfg.edges) == 6)
check("valid config layers", len(cfg.layers) == 7)

try:
    EnvConfig(num_nodes=3, edges=[(0,1),(0,2)], layers=[0,1], H=[1,2,3], B=[1,2,3])
    check("invalid layers length", False)
except ValueError as e:
    check("invalid layers length", "layers length" in str(e))

try:
    EnvConfig(num_nodes=3, edges=[(0,5)], layers=[0,1,1], H=[1,2,3], B=[1,2,3])
    check("edge out of range", False)
except ValueError as e:
    check("edge out of range", "out of range" in str(e))

try:
    EnvConfig(num_nodes=3, edges=[(1,1)], layers=[0,1,1], H=[1,2,3], B=[1,2,3])
    check("self-loop", False)
except ValueError as e:
    check("self-loop", "Self-loop" in str(e))


# ===== Network =====
print("\n=== Network Tests ===")

cfg2 = make_2layer_cfg()
up, down, topo, terms = build_network(cfg2)
check("2layer upstream[0]==[]", up[0] == [])
check("2layer downstream[0]", sorted(down[0]) == [1, 2])
check("2layer upstream[1]", up[1] == [0])
check("2layer terminals", terms == [1, 2])

cfg3 = make_3layer_cfg()
up3, down3, topo3, terms3 = build_network(cfg3)
check("3layer upstream[0]==[]", up3[0] == [])
check("3layer upstream[3]", up3[3] == [1])
check("3layer terminals", sorted(terms3) == [3, 4, 5, 6])

pos = {n: i for i, n in enumerate(topo3)}
check("3layer topo: 0<1", pos[0] < pos[1])
check("3layer topo: 1<3", pos[1] < pos[3])

# Multi-upstream
cfg_m = EnvConfig(
    num_nodes=4, edges=[(0, 2), (1, 2), (2, 3)],
    layers=[0, 0, 1, 2], H=[1,1,2,3], B=[5,5,4,3],
)
up_m, _, _, terms_m = build_network(cfg_m)
check("multi-upstream node 2", sorted(up_m[2]) == [0, 1])
check("multi-upstream terminals", terms_m == [3])


# ===== Demand =====
print("\n=== Demand Tests ===")

gen = DemandGenerator(make_3layer_cfg())
d = gen.generate(4)
check("poisson shape", d.shape == (4,))
check("poisson non-negative", (d >= 0).all())

gen1 = DemandGenerator(make_3layer_cfg())
gen2 = DemandGenerator(make_3layer_cfg())
d1 = gen1.generate(4)
d2 = gen2.generate(4)
check("deterministic with seed", np.array_equal(d1, d2))

# Normal
cfg_norm = EnvConfig(
    num_nodes=3, edges=[(0,1),(0,2)], layers=[0,1,1],
    H=[1,2,3], B=[5,4,3], demand_type="normal", demand_std=0.01, seed=42,
)
gen_norm = DemandGenerator(cfg_norm)
all_non_neg = all((gen_norm.generate(4) >= 0).all() for _ in range(100))
check("normal non-negative", all_non_neg)

# Merton
cfg_mt = EnvConfig(
    num_nodes=3, edges=[(0,1),(0,2)], layers=[0,1,1],
    H=[1,2,3], B=[5,4,3], demand_type="merton", seed=42,
)
gen_mt = DemandGenerator(cfg_mt)
d_mt = gen_mt.generate(4)
check("merton shape", d_mt.shape == (4,))


# ===== Allocation =====
print("\n=== Allocation Tests ===")

check("single upstream sufficient", allocate(10, [1.0], [100.0]) == [10.0])
check("single upstream insufficient", allocate(10, [1.0], [5.0]) == [5.0])

s = allocate(10, [0.6, 0.4], [100.0, 100.0])
check("two upstream sufficient 0", abs(s[0] - 6.0) < 0.01)
check("two upstream sufficient 1", abs(s[1] - 4.0) < 0.01)
check("two upstream sufficient sum", abs(sum(s) - 10.0) < 0.01)

s2 = allocate(10, [0.6, 0.4], [100.0, 2.0])
check("two upstream one short 1", abs(s2[1] - 2.0) < 0.01)
check("two upstream one short 0", abs(s2[0] - 8.0) < 0.01)

s3 = allocate(10, [0.5, 0.5], [3.0, 3.0])
check("total <= q", sum(s3) <= 10.0 + 1e-6)

check("zero order", allocate(0, [1.0], [100.0]) == [0.0])


# ===== Reward =====
print("\n=== Reward Tests ===")

inv = np.array([10.0, 5.0])
bl = np.array([2.0, 0.0])
H = np.array([1.0, 2.0])
B = np.array([5.0, 4.0])
rw, hc, bc = compute_reward(inv, bl, np.array([3.0, 2.0]), H, B, alpha=0.5)

check("holding costs", np.allclose(hc, [10.0, 10.0]))
check("backlog costs", np.allclose(bc, [10.0, 0.0]))
r_local = np.array([-20.0, -10.0])
r_global = np.full(2, r_local.mean())
expected = 0.5 * r_local + 0.5 * r_global
check("mixed reward", np.allclose(rw, expected))

rw0, _, _ = compute_reward(np.array([10.0, 0.0]), np.array([0.0, 5.0]),
                            np.array([2.0, 3.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), alpha=0.0)
check("alpha=0 global", rw0[0] == rw0[1])

rw1, _, _ = compute_reward(np.array([10.0, 0.0]), np.array([0.0, 5.0]),
                            np.array([2.0, 3.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), alpha=1.0)
check("alpha=1 local 0", rw1[0] == -10.0)
check("alpha=1 local 1", rw1[1] == -5.0)


# ===== Env =====
print("\n=== Env Tests ===")

env2 = MEIRPEnv(make_2layer_cfg())
obs, info = env2.reset()
check("reset obs shape", obs.shape == (3, 12))
check("reset inventory", env2.inventory[0] == 20.0)
check("reset backlog", env2.backlog[0] == 0.0)

# Step with BDQ actions: root -> int, non-root -> list[int]
actions = [5, [5], [5]]
obs, rewards, terminated, truncated, info = env2.step(actions)
check("step obs shape", obs.shape == (3, 12))
check("step rewards shape", rewards.shape == (3,))
check("step info has holding", "holding_costs" in info)
check("step info has backlog", "backlog_costs" in info)
check("step info has fill", "fill_rates" in info)

# Pipeline shift
env2.reset()
old_pipeline_last = env2.pipeline[0, -1]
actions2 = [10, [10], [10]]
env2.step(actions2)
check("pipeline shift: top node gets external", env2.pipeline[0, -1] == 10.0)

# Episode termination
env3 = MEIRPEnv(EnvConfig(
    num_nodes=2, edges=[(0,1)], layers=[0,1],
    H=[1,2], B=[5,4], episode_len=3, seed=42,
))
env3.reset()
actions3 = [5, [5]]
for _ in range(2):
    env3.step(actions3)
_, _, term, _, _ = env3.step(actions3)
check("episode terminates", term)

# Demand propagation
env_dp = MEIRPEnv(make_2layer_cfg())
env_dp.reset(seed=123)
actions_dp = [0, [6], [4]]
env_dp.step(actions_dp)
check("demand propagation: node 0", abs(env_dp.demand[0] - 10.0) < 0.01)

# 3-layer full episode
env_3l = MEIRPEnv(make_3layer_cfg())
env_3l.reset()
# BDQ format: root -> int, non-root -> list[int]
actions_3l = [5, [5], [5], [5], [5], [5], [5]]
for _ in range(env_3l.cfg.episode_len):
    obs, _, term, _, _ = env_3l.step(actions_3l)
    if term:
        break
check("3layer obs shape", obs.shape == (7, 12))

# Obs normalization
check("obs in [-1,1]", (obs >= -1.0).all() and (obs <= 1.0).all())

# Inventory decreases on demand
env_dec = MEIRPEnv(EnvConfig(
    num_nodes=2, edges=[(0,1)], layers=[0,1],
    H=[1,2], B=[5,4], lead_time=1,
    init_inventory=100.0, init_pipeline=0.0,
    demand_mean=20.0, seed=42,
))
env_dec.reset()
init_inv = env_dec.inventory[1]
actions_dec = [0, [0]]
for _ in range(5):
    env_dec.step(actions_dec)
check("inventory decreases or backlog grows",
       env_dec.inventory[1] < init_inv or env_dec.backlog[1] > 0)

# ===== Summary =====
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
if failed > 0:
    sys.exit(1)
else:
    print("All tests passed!")
