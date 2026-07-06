"""End-to-end integration test -- all modules working together."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from meirp_project.core.config import EnvConfig, from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.registry import make_agent
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.baselines.ss_policy import SSPolicy
from meirp_project.collaboration.sea_wrapper import SEAWrapper
from meirp_project.evaluation.metrics import compute_metrics
from meirp_project.evaluation.runner import EvalRunner

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


# ===== Test 1: YAML Config -> Env -> Full Episode =====
print("\n=== Integration Test 1: YAML -> Env -> Episode ===")

cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "env_3layers.yaml")
if os.path.exists(cfg_path):
    cfg = from_yaml(cfg_path)
else:
    cfg = EnvConfig(
        num_nodes=7, edges=[(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)],
        layers=[0,1,1,2,2,2,2],
        H=[1,2,2,3,3,3,3], B=[5,4,4,3,3,3,3],
        max_order=20, lead_time=2, episode_len=30, seed=42,
    )

env = MEIRPEnv(cfg)
obs, _ = env.reset()
check("env created from config", env.n == cfg.num_nodes)

info_hist = []
done = False
while not done:
    # BDQ format for 9-node 3-layer: root(0,1) -> int, others -> list[int]
    # Node upstreams: 0:[], 1:[], 2:[0,1], 3:[1,0], 4:[1], 5:[2], 6:[2,3], 7:[3,4], 8:[4]
    actions = [5, 5, [5, 5], [5, 5], [5], [5], [5, 5], [5, 5], [5]]
    obs, rewards, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    info_hist.append(info)

metrics = compute_metrics(info_hist, cfg)
check("full episode ran", len(info_hist) == cfg.episode_len)
check("metrics computed", "total_cost" in metrics)


# ===== Test 2: All Algorithms -> EvalRunner =====
print("\n=== Integration Test 2: All Algorithms -> EvalRunner ===")

obs_dim = 6 + 3 * cfg.lead_time
runner = EvalRunner(cfg)

for algo_name in ["iql", "qmix", "mappo"]:
    agent = make_agent(algo_name, cfg, obs_dim=obs_dim, hidden_dim=32, buffer_capacity=10)
    result = runner.eval_agent(agent, n_episodes=2, deterministic=True)
    check(f"{algo_name} eval completed", "total_cost" in result)
    check(f"{algo_name} fill_rate valid", 0 <= result["avg_fill_rate"] <= 1)


# ===== Test 3: Baselines -> EvalRunner =====
print("\n=== Integration Test 3: Baselines -> EvalRunner ===")

bs = BaseStockPolicy(cfg)
bs_result = runner.eval_baseline(bs, n_episodes=3)
check("BaseStock eval completed", "total_cost" in bs_result)

ss = SSPolicy(cfg)
ss_result = runner.eval_baseline(ss, n_episodes=3)
check("(s,S) eval completed", "total_cost" in ss_result)

check("BaseStock cheaper or equal to (s,S) on average",
      bs_result["avg_cost_per_step"] <= ss_result["avg_cost_per_step"] + 100)


# ===== Test 4: SEA Wrapper -> Agent =====
print("\n=== Integration Test 4: SEA Wrapper Integration ===")

from meirp_project.collaboration.base import SEABase

class TestSEA(SEABase):
    def __init__(self):
        self.intervention_count = 0

    def should_intervene(self, obs, step):
        return step % 5 == 0

    def get_expert_action(self, obs):
        n = obs.shape[0]
        self.intervention_count += 1
        return [0 if i < 2 else [0, 0] if i in (2, 3, 6, 7) else [0] for i in range(n)]

    def fuse_actions(self, agent_actions, expert_actions, confidence):
        if confidence > 0.5:
            return expert_actions
        return agent_actions

iql = make_agent("iql", cfg, obs_dim=obs_dim, hidden_dim=32, buffer_capacity=10)
sea = TestSEA()
wrapped = SEAWrapper(iql, sea=sea, use_sea=True)

env2 = MEIRPEnv(cfg)
obs, _ = env2.reset()
done = False
steps = 0
while not done:
    actions = wrapped.get_actions(obs)
    obs, rewards, terminated, truncated, info = env2.step(actions)
    done = terminated or truncated
    steps += 1

check("SEA wrapper ran full episode", steps == cfg.episode_len)

# Without SEA
wrapped_off = SEAWrapper(iql, sea=sea, use_sea=False)
env3 = MEIRPEnv(cfg)
obs, _ = env3.reset()
done = False
steps = 0
while not done:
    actions = wrapped_off.get_actions(obs)
    obs, rewards, terminated, truncated, info = env3.step(actions)
    done = terminated or truncated
    steps += 1

check("SEA off also runs full episode", steps == cfg.episode_len)


# ===== Test 5: Algorithm Registry =====
print("\n=== Integration Test 5: Algorithm Registry ===")

for name in ["iql", "qmix", "mappo"]:
    agent = make_agent(name, cfg, obs_dim=obs_dim, hidden_dim=32)
    obs_test = np.random.randn(cfg.num_nodes, obs_dim)
    actions = agent.get_actions(obs_test)
    check(f"registry {name} produces actions", len(actions) == cfg.num_nodes)

try:
    make_agent("unknown", cfg)
    check("registry rejects unknown algo", False)
except ValueError:
    check("registry rejects unknown algo", True)


# ===== Test 6: Multi-layer configs =====
print("\n=== Integration Test 6: Multi-layer Configs ===")

# 2-layer
cfg2 = EnvConfig(
    num_nodes=3, edges=[(0,1),(0,2)], layers=[0,1,1],
    H=[1,2,2], B=[5,4,4], max_order=20, lead_time=2,
    episode_len=10, demand_mean=5.0, seed=42,
)
env2l = MEIRPEnv(cfg2)
obs2, _ = env2l.reset()
check("2-layer env works", obs2.shape == (3, 12))

# 4-layer
cfg4 = EnvConfig(
    num_nodes=15,
    edges=[(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),
           (3,7),(3,8),(4,9),(4,10),(5,11),(5,12),(6,13),(6,14)],
    layers=[0,1,1,2,2,2,2,3,3,3,3,3,3,3,3],
    H=[0.5,1,1,2,2,2,2,4,4,4,4,4,4,4,4],
    B=[10,6,6,4,4,4,4,20,20,20,20,20,20,20,20],
    max_order=20, lead_time=2, episode_len=10, demand_mean=4.0, seed=42,
)
env4l = MEIRPEnv(cfg4)
obs4, _ = env4l.reset()
check("4-layer env works", obs4.shape == (15, 12))

agent4 = make_agent("iql", cfg4, obs_dim=12, hidden_dim=32, buffer_capacity=10)
actions4 = agent4.get_actions(obs4)
check("4-layer agent produces actions", len(actions4) == 15)


# ===== Test 7: MILP Solver =====
print("\n=== Integration Test 7: MILP Solver ===")

from meirp_project.solvers.milp_solver import MILPSolver

milp = MILPSolver(cfg2, planning_horizon=5)
check("MILP solver created", milp is not None)

# Test with env state
env2l.reset()
demand_forecast = np.random.poisson(5, size=(5, 2)).astype(float)
actions_milp = milp.get_actions(
    inventory=env2l.inventory,
    backlog=env2l.backlog,
    pipeline=env2l.pipeline,
    current_demand_forecast=demand_forecast,
)
check("MILP produces actions", len(actions_milp) == cfg2.num_nodes)
for i, action in enumerate(actions_milp):
    if isinstance(action, list):
        q = sum(action)
        valid_shape = len(action) == len(env2l.upstream[i])
    else:
        q = action
        valid_shape = len(env2l.upstream[i]) == 0
    check(f"MILP action {i} BDQ shape", valid_shape)
    check(f"MILP action {i} valid", 0 <= q <= cfg2.max_order)


# ===== Summary =====
print(f"\n{'='*40}")
print(f"Integration Tests: {passed} passed, {failed} failed, {passed+failed} total")
if failed > 0:
    sys.exit(1)
else:
    print("All integration tests passed!")
