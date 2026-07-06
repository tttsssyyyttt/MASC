"""Test IQL algorithm: instantiation, action selection, learning loop."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from meirp_project.core.config import EnvConfig
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.iql.agent import IQLAgent
from meirp_project.algorithms.iql.buffer import ReplayBuffer

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
        max_order=10, lead_time=2, episode_len=20,
        demand_mean=5.0, seed=42,
    )


# ===== Buffer Tests =====
print("\n=== ReplayBuffer Tests ===")

buf = ReplayBuffer(capacity=100)
check("buffer empty", len(buf) == 0)

obs = np.random.randn(3, 12)
# BDQ format: root -> int, non-root -> {"q": total, "alpha": weights}
actions = [3, {"q": 5, "alpha": [1]}, {"q": 2, "alpha": [1]}]
rewards = np.array([-1.0, -2.0, -3.0])
next_obs = np.random.randn(3, 12)
dones = np.array([0.0, 0.0, 0.0])

buf.push(obs, actions, rewards, next_obs, dones)
check("buffer size 1", len(buf) == 1)

for _ in range(9):
    buf.push(obs, actions, rewards, next_obs, dones)
check("buffer size 10", len(buf) == 10)

batch = buf.sample(4)
check("sample obs shape", batch["obs"].shape == (4, 3, 12))
check("sample rewards shape", batch["rewards"].shape == (4, 3))


# ===== IQLAgent Tests =====
print("\n=== IQLAgent Tests ===")

cfg = make_cfg()
env = MEIRPEnv(cfg)
obs_dim = 6 + 3 * cfg.lead_time  # = 12

agent = IQLAgent(
    cfg=cfg,
    obs_dim=obs_dim,
    hidden_dim=64,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=1000,
    buffer_capacity=1000,
    batch_size=8,
    target_update_freq=50,
)

check("agent created", agent is not None)
check("n q_nets", len(agent.q_nets) == cfg.num_nodes)
check("n target_nets", len(agent.target_nets) == cfg.num_nodes)

# Test get_actions (BDQ format)
obs, _ = env.reset()
actions = agent.get_actions(obs)
check("actions length", len(actions) == cfg.num_nodes)

# Node 0: root (no upstream) -> int
check("node 0 is root -> int", isinstance(actions[0], (int, np.integer)))
check("node 0 q in range", 0 <= actions[0] <= cfg.max_order)
# Node 1: non-root, 1 upstream -> q + alpha
check("node 1 is non-root -> dict", isinstance(actions[1], dict))
check("node 1 alpha len 1", len(actions[1].get("alpha", [])) == 1)
check("node 1 q in range", 0 <= actions[1].get("q", -1) <= cfg.max_order)
check("node 1 alpha in range", all(0 <= q <= cfg.max_order for q in actions[1].get("alpha", [])))
# Node 2: non-root, 1 upstream -> q + alpha
check("node 2 is non-root -> dict", isinstance(actions[2], dict))
check("node 2 alpha len 1", len(actions[2].get("alpha", [])) == 1)
check("node 2 q in range", 0 <= actions[2].get("q", -1) <= cfg.max_order)
check("node 2 alpha in range", all(0 <= q <= cfg.max_order for q in actions[2].get("alpha", [])))

# Test deterministic actions (epsilon=0)
actions_det = agent.get_actions(obs, deterministic=True)
for i in range(cfg.num_nodes):
    if isinstance(actions_det[i], dict):
        q_val = actions_det[i].get("q", -1)
    elif isinstance(actions_det[i], list):
        q_val = actions_det[i][0]
    else:
        q_val = actions_det[i]
    check(f"deterministic q in range (node {i})", 0 <= q_val <= cfg.max_order)

# Test epsilon schedule
agent.total_steps = 0
eps0 = agent._get_epsilon()
check("epsilon start ~1.0", abs(eps0 - 1.0) < 0.01)

agent.total_steps = 1000
eps1 = agent._get_epsilon()
check("epsilon decayed", eps1 < eps0)

agent.total_steps = 100000
eps_final = agent._get_epsilon()
check("epsilon at floor", abs(eps_final - 0.05) < 0.01)


# ===== Learning Loop Test =====
print("\n=== IQL Learning Test ===")

# Run a short training loop
agent2 = IQLAgent(
    cfg=cfg,
    obs_dim=obs_dim,
    hidden_dim=32,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=500,
    buffer_capacity=2000,
    batch_size=8,
    target_update_freq=20,
)

env2 = MEIRPEnv(cfg)
n_episodes = 5
ep_rewards = []

for ep in range(n_episodes):
    obs, _ = env2.reset()
    ep_reward = 0.0
    done = False
    steps = 0

    while not done:
        actions = agent2.get_actions(obs)
        next_obs, rewards, terminated, truncated, info = env2.step(actions)
        done = terminated or truncated

        dones_arr = np.full(cfg.num_nodes, float(terminated))
        agent2.store_transition(obs, actions, rewards, next_obs, dones_arr)

        # Update if enough data
        if len(agent2.buffer) >= 8:
            metrics = agent2.update()

        ep_reward += rewards.sum()
        obs = next_obs
        steps += 1

    ep_rewards.append(ep_reward)
    check(f"episode {ep+1} completed", True)

# Verify learning produces finite rewards
check("all rewards finite", all(np.isfinite(r) for r in ep_rewards))
check("training ran without errors", True)


# ===== Save/Load Test =====
print("\n=== IQL Save/Load Test ===")

import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, "iql_test.pt")

    # Get actions before save
    obs, _ = env2.reset()
    actions_before = agent2.get_actions(obs, deterministic=True)

    # Save
    agent2.save(save_path)
    check("save succeeded", os.path.exists(save_path))

    # Load into new agent
    agent3 = IQLAgent(
        cfg=cfg, obs_dim=obs_dim, hidden_dim=32,
        lr=1e-3, buffer_capacity=10,
    )
    agent3.load(save_path)

    # Get actions after load
    actions_after = agent3.get_actions(obs, deterministic=True)

    check("same actions after load", actions_before == actions_after)


# ===== Summary =====
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
if failed > 0:
    sys.exit(1)
else:
    print("All tests passed!")
