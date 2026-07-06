"""Test MAPPO algorithm: actor, critic, buffer, learning loop."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from meirp_project.core.config import EnvConfig
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.mappo.agent import MAPPOAgent, BranchingActor, Critic
from meirp_project.algorithms.mappo.buffer import RolloutBuffer

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


# ===== Actor/Critic Tests =====
print("\n=== Actor/Critic Tests ===")

obs_dim = 12
max_order = 10

actor = BranchingActor(obs_dim, n_branches=2, max_order=max_order, hidden_dim=64)
critic = Critic(obs_dim, hidden_dim=64)

obs_t = torch.randn(4, obs_dim)
state_t = torch.randn(4, obs_dim)

# BranchingActor returns list of logit tensors (one per branch)
logits_list = actor(obs_t)
check("actor output is list", isinstance(logits_list, list))
check("actor logits shape", logits_list[0].shape == (4, max_order + 1))

probs = torch.softmax(logits_list[0], dim=-1)
check("actor probs sum to 1", torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5))

value = critic(state_t)
check("critic output shape", value.shape == (4, 1))


# ===== RolloutBuffer Tests =====
print("\n=== RolloutBuffer Tests ===")

buf = RolloutBuffer()
check("buffer empty", len(buf) == 0)

obs = np.random.randn(3, 12)
# BDQ format: root -> int, non-root -> {"q": total, "alpha": weights}
actions = [3, {"q": 5, "alpha": [1]}, {"q": 2, "alpha": [1]}]
rewards = np.array([-1.0, -2.0, -3.0])
values = np.array([0.5, -0.5, 1.0])
log_probs = np.array([-2.1, -1.8, -2.5])
dones = np.array([0.0, 0.0, 0.0])

for _ in range(10):
    buf.push(obs, actions, rewards, values, log_probs, dones)

check("buffer size 10", len(buf) == 10)

buf.compute_returns(gamma=0.99, lam=0.95)
check("advantages computed", hasattr(buf, 'advantages'))
check("returns computed", hasattr(buf, 'returns'))
check("advantages shape", buf.advantages.shape == (10, 3))
check("returns shape", buf.returns.shape == (10, 3))

# Test get_batches
n_batches = 0
for batch in buf.get_batches(batch_size=8, n_agents=3, obs_dim=12):
    check("batch obs shape", batch["obs"].shape[1:] == (3, obs_dim))
    check("batch has agent_indices", "agent_indices" in batch)
    check("batch has actions_q_per_branch", "actions_q_per_branch" in batch)
    n_batches += 1
    break  # Just test first batch

buf.clear()
check("buffer cleared", len(buf) == 0)


# ===== MAPPOAgent Tests =====
print("\n=== MAPPOAgent Tests ===")

cfg = make_cfg()
env = MEIRPEnv(cfg)

agent = MAPPOAgent(
    cfg=cfg, obs_dim=obs_dim,
    hidden_dim=64, lr_actor=3e-4, lr_critic=1e-3,
    gamma=0.99, gae_lambda=0.95,
    clip_epsilon=0.2, epochs_per_update=3,
    mini_batch_size=8, entropy_coef=0.01,
)

check("agent created", agent is not None)
check("actors count", len(agent.actors) == cfg.num_nodes)
check("critic exists", agent.critic is not None)

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

# Test deterministic
actions_det = agent.get_actions(obs, deterministic=True)
check("det node 0 q in range", 0 <= actions_det[0] <= cfg.max_order)
check("det node 1 q in range", 0 <= actions_det[1].get("q", -1) <= cfg.max_order)

# Check internal values stored
check("last values stored", hasattr(agent, '_last_values'))
check("last log_probs stored", hasattr(agent, '_last_log_probs'))


# ===== Learning Loop =====
print("\n=== MAPPO Learning Test ===")

agent2 = MAPPOAgent(
    cfg=cfg, obs_dim=obs_dim,
    hidden_dim=32, lr_actor=3e-4, lr_critic=1e-3,
    epochs_per_update=3, mini_batch_size=8,
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

        ep_reward += rewards.sum()
        obs = next_obs

    # Update after each episode
    metrics = agent2.update()
    ep_rewards.append(ep_reward)
    check(f"mappo episode {ep+1} completed", True)

check("all mappo rewards finite", all(np.isfinite(r) for r in ep_rewards))
check("update produced metrics", "policy_loss" in metrics or "loss" in metrics)


# ===== Save/Load =====
print("\n=== MAPPO Save/Load Test ===")

import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, "mappo_test.pt")
    actions_before = agent2.get_actions(obs, deterministic=True)
    agent2.save(save_path)
    check("mappo save succeeded", os.path.exists(save_path))

    agent3 = MAPPOAgent(cfg=cfg, obs_dim=obs_dim, hidden_dim=32)
    agent3.load(save_path)
    actions_after = agent3.get_actions(obs, deterministic=True)

    check("same actions after mappo load", actions_before == actions_after)


# ===== Summary =====
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
if failed > 0:
    sys.exit(1)
else:
    print("All tests passed!")
