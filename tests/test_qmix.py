"""Test QMIX algorithm: mixer, agent, learning loop."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from meirp_project.core.config import EnvConfig
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.qmix.agent import QMIXAgent
from meirp_project.algorithms.qmix.mixer import QMixer

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


# ===== Mixer Tests =====
print("\n=== QMixer Tests ===")

n_agents = 3
state_dim = 36  # 3 agents * 12 obs_dim
mixer = QMixer(n_agents, state_dim, hidden_dim=32)

q_vals = torch.randn(4, n_agents)  # batch=4
states = torch.randn(4, state_dim)

q_tot = mixer(q_vals, states)
check("mixer output shape", q_tot.shape == (4, 1))

# Test monotonicity: increasing any Q_i should not decrease Q_tot
q_vals_2 = q_vals.clone()
q_vals_2[0, 0] += 1.0  # increase Q for agent 0
q_tot_2 = mixer(q_vals_2, states)
check("monotonicity: Q_tot increases when Q_i increases", q_tot_2[0, 0] > q_tot[0, 0] - 1e-6)


# ===== QMIXAgent Tests =====
print("\n=== QMIXAgent Tests ===")

cfg = make_cfg()
env = MEIRPEnv(cfg)
obs_dim = 6 + 3 * cfg.lead_time

agent = QMIXAgent(
    cfg=cfg, obs_dim=obs_dim,
    hidden_dim=64, mixer_hidden=32,
    lr=5e-4, gamma=0.99,
    buffer_capacity=1000, batch_size=8,
    target_update_freq=50,
)

check("agent created", agent is not None)
check("mixer exists", agent.mixer is not None)
check("target_mixer exists", agent.target_mixer is not None)

# Test get_actions (BDQ format)
obs, _ = env.reset()
actions = agent.get_actions(obs)
check("actions length", len(actions) == cfg.num_nodes)
# Node 0: root -> int; Node 1,2: non-root -> q + alpha
check("node 0 is root -> int", isinstance(actions[0], (int, np.integer)))
check("node 0 q in range", 0 <= actions[0] <= cfg.max_order)
check("node 1 is non-root -> dict", isinstance(actions[1], dict))
check("node 1 q in range", 0 <= actions[1].get("q", -1) <= cfg.max_order)
check("node 1 alpha len 1", len(actions[1].get("alpha", [])) == 1)
check("node 1 alpha in range", all(0 <= q <= cfg.max_order for q in actions[1].get("alpha", [])))
check("node 2 is non-root -> dict", isinstance(actions[2], dict))
check("node 2 q in range", 0 <= actions[2].get("q", -1) <= cfg.max_order)
check("node 2 alpha len 1", len(actions[2].get("alpha", [])) == 1)
check("node 2 alpha in range", all(0 <= q <= cfg.max_order for q in actions[2].get("alpha", [])))

# Test deterministic
actions_det = agent.get_actions(obs, deterministic=True)
check("deterministic node 1 q in range", 0 <= actions_det[1].get("q", -1) <= cfg.max_order)


# ===== Learning Loop =====
print("\n=== QMIX Learning Test ===")

agent2 = QMIXAgent(
    cfg=cfg, obs_dim=obs_dim,
    hidden_dim=32, mixer_hidden=16,
    lr=5e-4, buffer_capacity=2000, batch_size=8,
    target_update_freq=20,
)

env2 = MEIRPEnv(cfg)
ep_rewards = []

for ep in range(5):
    obs, _ = env2.reset()
    ep_reward = 0.0
    done = False

    while not done:
        actions = agent2.get_actions(obs)
        next_obs, rewards, terminated, truncated, info = env2.step(actions)
        done = terminated or truncated

        dones_arr = np.full(cfg.num_nodes, float(terminated))
        agent2.store_transition(obs, actions, rewards, next_obs, dones_arr)

        if len(agent2.buffer) >= 8:
            metrics = agent2.update()

        ep_reward += rewards.sum()
        obs = next_obs

    ep_rewards.append(ep_reward)
    check(f"qmix episode {ep+1} completed", True)

check("all qmix rewards finite", all(np.isfinite(r) for r in ep_rewards))


# ===== Save/Load =====
print("\n=== QMIX Save/Load Test ===")

import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, "qmix_test.pt")
    actions_before = agent2.get_actions(obs, deterministic=True)
    agent2.save(save_path)
    check("qmix save succeeded", os.path.exists(save_path))

    agent3 = QMIXAgent(cfg=cfg, obs_dim=obs_dim, hidden_dim=32, mixer_hidden=16, buffer_capacity=10)
    agent3.load(save_path)
    actions_after = agent3.get_actions(obs, deterministic=True)

    check("same actions after qmix load", actions_before == actions_after)


# ===== Summary =====
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
if failed > 0:
    sys.exit(1)
else:
    print("All tests passed!")
