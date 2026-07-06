"""Smoke-test MADRL agents with the GNN/GAT encoder enabled."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meirp_project.algorithms.registry import make_agent
from meirp_project.core.config import EnvConfig
from meirp_project.core.env import MEIRPEnv
from meirp_project.core.network import build_network


def make_cfg():
    return EnvConfig(
        num_nodes=5,
        edges=[(0, 1), (0, 2), (1, 3), (2, 3), (2, 4)],
        layers=[0, 1, 1, 2, 2],
        H=[1, 2, 2, 3, 3],
        B=[5, 4, 4, 3, 3],
        max_order=8,
        lead_time=2,
        episode_len=6,
        demand_mean=4.0,
        seed=7,
    )


def check_action_bounds(actions, cfg):
    upstream, _, _, _ = build_network(cfg)
    for i, action in enumerate(actions):
        if isinstance(action, dict):
            assert 0 <= int(action.get("q", 0)) <= cfg.max_order
            alpha = action.get("alpha", [])
            assert len(alpha) == len(upstream[i])
            assert all(0 <= int(q) <= cfg.max_order for q in alpha)
        elif isinstance(action, list):
            assert all(0 <= int(q) <= cfg.max_order for q in action)
        else:
            assert 0 <= int(action) <= cfg.max_order


def run_short_training(algo):
    cfg = make_cfg()
    env = MEIRPEnv(cfg)
    obs_dim = 6 + 3 * cfg.lead_time
    kwargs = {"hidden_dim": 16}
    if algo in {"iql", "qmix"}:
        kwargs.update({
            "batch_size": 4,
            "buffer_capacity": 100,
            "epsilon_decay": 20,
            "target_update_freq": 5,
        })
    if algo == "qmix":
        kwargs["mixer_hidden"] = 8
    if algo == "mappo":
        kwargs.update({
            "mini_batch_size": 4,
            "epochs_per_update": 1,
        })

    agent = make_agent(algo, cfg, obs_dim=obs_dim, device="cpu", use_gnn=True, **kwargs)
    rewards = []

    for ep in range(2):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            actions = agent.get_actions(obs)
            check_action_bounds(actions, cfg)
            next_obs, reward, term, trunc, _ = env.step(actions)
            done = term or trunc
            agent.store_transition(
                obs,
                actions,
                reward,
                next_obs,
                np.full(cfg.num_nodes, float(term)),
            )
            if algo in {"iql", "qmix"} and len(agent.buffer) >= agent.batch_size:
                metrics = agent.update()
                assert all(np.isfinite(v) for v in metrics.values())
            ep_reward += float(reward.sum())
            obs = next_obs
        if algo == "mappo":
            metrics = agent.update()
            assert all(np.isfinite(v) for v in metrics.values())
        rewards.append(ep_reward)

    assert all(np.isfinite(r) for r in rewards)
    assert getattr(agent, "use_gnn", False)
    assert hasattr(agent.encoder, "gat")
    assert agent.encoder.gat.get_attention_weights() is not None


def test_iql_gnn_short_training():
    run_short_training("iql")


def test_qmix_gnn_short_training():
    run_short_training("qmix")


def test_mappo_gnn_short_training():
    run_short_training("mappo")


if __name__ == "__main__":
    for name, fn in [
        ("IQL-GNN", test_iql_gnn_short_training),
        ("QMIX-GNN", test_qmix_gnn_short_training),
        ("MAPPO-GNN", test_mappo_gnn_short_training),
    ]:
        fn()
        print(f"[PASS] {name} short training")
    print("All MADRL+GNN/GAT smoke tests passed.")
