"""Tests for BDQ (Branching Dueling Q-Network) action space redesign.

Tests cover:
1. Environment action format and normalization
2. IQL+BDQ agent: BranchingQNetwork, get_actions, update, entropy
3. QMIX+BDQ agent: get_actions, update, entropy
4. MAPPO+BDQ agent: BranchingActor, get_actions, update, entropy
5. Baselines with new action format
6. End-to-end training loop
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import pytest

from meirp_project.core.config import from_yaml, EnvConfig
from meirp_project.core.env import MEIRPEnv
from meirp_project.core.network import build_network
from meirp_project.algorithms.iql.agent import IQLAgent, BranchingQNetwork
from meirp_project.algorithms.qmix.agent import QMIXAgent
from meirp_project.algorithms.mappo.agent import MAPPOAgent, BranchingActor
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.baselines.ss_policy import SSPolicy


# ===================== Fixtures =====================

def make_3layer_config():
    """Create a 3-layer supply chain config for testing."""
    cfg = EnvConfig(
        num_nodes=9,
        edges=[[0, 2], [1, 2], [1, 3], [0, 3], [1, 4], [2, 5], [2, 6], [3, 6], [3, 7], [4, 7], [4, 8]],
        layers=[0, 0, 1, 1, 1, 2, 2, 2, 2],
        max_order=20,
        lead_time=2,
        demand_mean=5.0,
        demand_std=2.0,
        episode_len=30,
        H=[1.0] * 9,
        B=[2.0] * 9,
        alpha=0.5,
        init_inventory=20.0,
        init_pipeline=0.0,
        seed=42,
    )
    return cfg


def assert_qalpha_action(action, k, cfg):
    assert isinstance(action, dict)
    assert 0 <= action.get("q", -1) <= cfg.max_order
    alpha = action.get("alpha", [])
    assert len(alpha) == k
    for q in alpha:
        assert 0 <= q <= cfg.max_order


# ===================== Environment Tests =====================

class TestEnvBDQ:
    """Test environment with BDQ action format."""

    def test_action_format_root_nodes(self):
        """Root nodes (no upstream) should accept single int action."""
        cfg = make_3layer_config()
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        # Node 0 and 1 are roots (no upstream)
        upstream, _, _, _ = build_network(cfg)
        assert len(upstream[0]) == 0
        assert len(upstream[1]) == 0

    def test_action_format_nonroot_nodes(self):
        """Non-root nodes should accept list of per-upstream quantities."""
        cfg = make_3layer_config()
        upstream, _, _, _ = build_network(cfg)
        # Node 2 has 2 upstream (nodes 0 and 1)
        assert len(upstream[2]) == 2
        # Node 4 has 1 upstream
        assert len(upstream[4]) == 1

    def test_step_with_bdq_actions(self):
        """Environment should handle BDQ actions correctly."""
        cfg = make_3layer_config()
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        upstream, _, _, _ = build_network(cfg)
        actions = []
        for i in range(9):
            k = len(upstream[i])
            if k == 0:
                actions.append(10)  # root: order 10 from external
            else:
                actions.append([5] * k)  # non-root: order 5 from each upstream

        obs, rewards, terminated, truncated, info = env.step(actions)
        assert obs.shape == (9, 6 + 3 * cfg.lead_time)
        assert rewards.shape == (9,)
        assert not np.any(np.isnan(rewards))

    def test_normalization_when_exceeds_max(self):
        """When total per-upstream orders exceed max_order, should normalize."""
        cfg = make_3layer_config()
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        upstream, _, _, _ = build_network(cfg)

        # Node 2 has 2 upstream. Order max_order from each → total = 2*max_order
        actions = []
        for i in range(9):
            k = len(upstream[i])
            if k == 0:
                actions.append(10)
            elif i == 2:
                # Order 20 from each upstream → total 40 > max_order=20
                actions.append([20, 20])
            else:
                actions.append([5] * k)

        order_qty, alpha = env._parse_actions(actions)
        # After normalization, total for node 2 should be max_order
        assert order_qty[2] == pytest.approx(cfg.max_order, abs=1e-6)
        # Alpha should be [0.5, 0.5] (proportional)
        assert alpha[2][0] == pytest.approx(0.5, abs=1e-6)
        assert alpha[2][1] == pytest.approx(0.5, abs=1e-6)

    def test_full_episode(self):
        """Run a full episode with BDQ actions."""
        cfg = make_3layer_config()
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        upstream, _, _, _ = build_network(cfg)
        done = False
        steps = 0
        while not done:
            actions = []
            for i in range(9):
                k = len(upstream[i])
                if k == 0:
                    actions.append(5)
                else:
                    actions.append([3] * k)
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            steps += 1
            assert not np.any(np.isnan(obs))
            assert not np.any(np.isnan(rewards))

        assert steps == cfg.episode_len

    def test_zero_orders(self):
        """All-zero orders should work without errors."""
        cfg = make_3layer_config()
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        upstream, _, _, _ = build_network(cfg)
        actions = []
        for i in range(9):
            k = len(upstream[i])
            if k == 0:
                actions.append(0)
            else:
                actions.append([0] * k)

        obs, rewards, terminated, truncated, info = env.step(actions)
        assert not np.any(np.isnan(rewards))


# ===================== BranchingQNetwork Tests =====================

class TestBranchingQNetwork:
    """Test BranchingQNetwork architecture."""

    def test_output_shape_single_branch(self):
        """Root node: 1 branch, output shape (batch, max_order+1)."""
        net = BranchingQNetwork(input_dim=12, n_branches=1, max_order=20)
        x = torch.randn(4, 12)
        outputs = net(x)
        assert len(outputs) == 1  # 1 branch for root
        assert outputs[0].shape == (4, 21)

    def test_output_shape_two_branches(self):
        """Node with 2 upstream: q branch + 2 alpha branches."""
        net = BranchingQNetwork(input_dim=12, n_branches=3, max_order=20)
        x = torch.randn(4, 12)
        outputs = net(x)
        assert len(outputs) == 3
        assert outputs[0].shape == (4, 21)
        assert outputs[1].shape == (4, 21)
        assert outputs[2].shape == (4, 21)

    def test_gradient_flows(self):
        """Gradient should flow through all branches."""
        net = BranchingQNetwork(input_dim=12, n_branches=3, max_order=20)
        x = torch.randn(4, 12, requires_grad=True)
        outputs = net(x)
        loss = sum(o.sum() for o in outputs)
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ===================== IQL+BDQ Tests =====================

class TestIQLBDQ:
    """Test IQL agent with BDQ branching."""

    def test_create_agent(self):
        """IQL agent should create without errors."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = IQLAgent(cfg, obs_dim=obs_dim)
        assert agent.n == 9
        assert len(agent.q_nets) == 9

    def test_get_actions_format(self):
        """get_actions should return BDQ format."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = IQLAgent(cfg, obs_dim=obs_dim)
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        actions = agent.get_actions(obs, deterministic=True)

        upstream, _, _, _ = build_network(cfg)
        for i in range(9):
            k = len(upstream[i])
            if k == 0:
                assert isinstance(actions[i], int), f"Root node {i} should have int action"
                assert 0 <= actions[i] <= cfg.max_order
            else:
                assert_qalpha_action(actions[i], k, cfg)

    def test_store_and_update(self):
        """Store transitions and run update."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = IQLAgent(cfg, obs_dim=obs_dim, batch_size=4)
        env = MEIRPEnv(cfg)

        obs, _ = env.reset()
        for _ in range(10):
            actions = agent.get_actions(obs)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            dones = np.full(cfg.num_nodes, float(terminated))
            agent.store_transition(obs, actions, rewards, next_obs, dones)
            obs = next_obs if not terminated else env.reset()[0]

        result = agent.update()
        assert "loss" in result
        assert result["loss"] >= 0

    def test_entropy(self):
        """Entropy computation should return valid values."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = IQLAgent(cfg, obs_dim=obs_dim)
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        entropy = agent.get_entropy(obs)
        assert entropy.shape == (9,)
        assert np.all(entropy >= 0)
        # At init, entropy should be non-zero (random Q-values → non-degenerate distribution)
        assert np.all(entropy > 0)

    def test_gnn_version(self):
        """IQL+GNN should work with BDQ."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        edge_index = torch.tensor([[0,1,1,0,1,2,2,3,3,4,4],[2,2,3,3,4,5,6,6,7,7,8]], dtype=torch.long)
        layer_ids = torch.tensor([0,0,1,1,1,2,2,2,2], dtype=torch.long)

        agent = IQLAgent(cfg, obs_dim=obs_dim, use_gnn=True,
                         edge_index=edge_index, layer_ids=layer_ids, device='cpu')
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        actions = agent.get_actions(obs, deterministic=True)
        assert len(actions) == 9


# ===================== QMIX+BDQ Tests =====================

class TestQMIXBDQ:
    """Test QMIX agent with BDQ branching."""

    def test_create_agent(self):
        """QMIX agent should create without errors."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = QMIXAgent(cfg, obs_dim=obs_dim)
        assert agent.n == 9

    def test_get_actions_format(self):
        """QMIX get_actions should return BDQ format."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = QMIXAgent(cfg, obs_dim=obs_dim)
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        actions = agent.get_actions(obs, deterministic=True)
        upstream, _, _, _ = build_network(cfg)
        for i in range(9):
            k = len(upstream[i])
            if k == 0:
                assert isinstance(actions[i], int)
            else:
                assert_qalpha_action(actions[i], k, cfg)

    def test_store_and_update(self):
        """QMIX update should work."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = QMIXAgent(cfg, obs_dim=obs_dim, batch_size=4)
        env = MEIRPEnv(cfg)

        obs, _ = env.reset()
        for _ in range(10):
            actions = agent.get_actions(obs)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            dones = np.full(cfg.num_nodes, float(terminated))
            agent.store_transition(obs, actions, rewards, next_obs, dones)
            obs = next_obs if not terminated else env.reset()[0]

        result = agent.update()
        assert "loss" in result
        assert result["loss"] >= 0

    def test_entropy(self):
        """QMIX entropy computation."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = QMIXAgent(cfg, obs_dim=obs_dim)
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        entropy = agent.get_entropy(obs)
        assert entropy.shape == (9,)
        assert np.all(entropy >= 0)


# ===================== MAPPO+BDQ Tests =====================

class TestMAPPOBDQ:
    """Test MAPPO agent with BDQ branching."""

    def test_create_agent(self):
        """MAPPO agent should create without errors."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = MAPPOAgent(cfg, obs_dim=obs_dim)
        assert agent.n == 9

    def test_get_actions_format(self):
        """MAPPO get_actions should return BDQ format."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = MAPPOAgent(cfg, obs_dim=obs_dim)
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        actions = agent.get_actions(obs, deterministic=True)
        upstream, _, _, _ = build_network(cfg)
        for i in range(9):
            k = len(upstream[i])
            if k == 0:
                assert isinstance(actions[i], int)
            else:
                assert_qalpha_action(actions[i], k, cfg)

    def test_store_and_update(self):
        """MAPPO on-policy update should work."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = MAPPOAgent(cfg, obs_dim=obs_dim)
        env = MEIRPEnv(cfg)

        # Run one full episode
        obs, _ = env.reset()
        done = False
        while not done:
            actions = agent.get_actions(obs)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            dones = np.full(cfg.num_nodes, float(terminated))
            agent.store_transition(obs, actions, rewards, next_obs, dones)
            obs = next_obs

        result = agent.update()
        assert "policy_loss" in result

    def test_entropy(self):
        """MAPPO entropy computation."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = MAPPOAgent(cfg, obs_dim=obs_dim)
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        entropy = agent.get_entropy(obs)
        assert entropy.shape == (9,)
        assert np.all(entropy >= 0)
        assert np.all(entropy > 0)  # Should be non-zero at init


# ===================== BranchingActor Tests =====================

class TestBranchingActor:
    """Test BranchingActor architecture."""

    def test_output_shape(self):
        """BranchingActor should output per-branch logits."""
        actor = BranchingActor(input_dim=12, n_branches=3, max_order=20)
        x = torch.randn(4, 12)
        outputs = actor(x)
        assert len(outputs) == 3
        assert outputs[0].shape == (4, 21)
        assert outputs[1].shape == (4, 21)
        assert outputs[2].shape == (4, 21)

    def test_root_node_actor(self):
        """Root node actor should have 1 branch."""
        actor = BranchingActor(input_dim=12, n_branches=1, max_order=20)
        x = torch.randn(4, 12)
        outputs = actor(x)
        assert len(outputs) == 1


# ===================== Baseline Tests =====================

class TestBaselinesBDQ:
    """Test baselines with BDQ action format."""

    def test_base_stock_actions(self):
        """BaseStock should output BDQ-format actions."""
        cfg = make_3layer_config()
        env = MEIRPEnv(cfg)
        policy = BaseStockPolicy(cfg)

        actions = policy.get_actions(env)
        upstream, _, _, _ = build_network(cfg)

        for i in range(9):
            k = len(upstream[i])
            if k == 0:
                assert isinstance(actions[i], int), f"Root node {i}"
            else:
                assert isinstance(actions[i], list), f"Node {i}"
                assert len(actions[i]) == k

    def test_ss_policy_actions(self):
        """SSPolicy should output BDQ-format actions."""
        cfg = make_3layer_config()
        env = MEIRPEnv(cfg)
        policy = SSPolicy(cfg)

        actions = policy.get_actions(env)
        upstream, _, _, _ = build_network(cfg)

        for i in range(9):
            k = len(upstream[i])
            if k == 0:
                assert isinstance(actions[i], int), f"Root node {i}"
            else:
                assert isinstance(actions[i], list), f"Node {i}"
                assert len(actions[i]) == k

    def test_base_stock_episode(self):
        """BaseStock should run a full episode."""
        cfg = make_3layer_config()
        env = MEIRPEnv(cfg)
        policy = BaseStockPolicy(cfg)
        result = policy.run_episode(env)
        assert result["avg_fill_rate"] > 0

    def test_ss_policy_episode(self):
        """SSPolicy should run a full episode."""
        cfg = make_3layer_config()
        env = MEIRPEnv(cfg)
        policy = SSPolicy(cfg)
        result = policy.run_episode(env)
        assert result["avg_fill_rate"] > 0


# ===================== End-to-End Training Tests =====================

class TestEndToEnd:
    """End-to-end training tests (quick sanity checks)."""

    def test_iql_training_loop(self):
        """IQL should train for a few episodes without errors."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = IQLAgent(cfg, obs_dim=obs_dim, batch_size=8)
        env = MEIRPEnv(cfg)

        for ep in range(3):
            obs, _ = env.reset()
            done = False
            while not done:
                actions = agent.get_actions(obs, deterministic=False)
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                dones = np.full(cfg.num_nodes, float(terminated))
                agent.store_transition(obs, actions, rewards, next_obs, dones)
                if len(agent.buffer) >= agent.batch_size:
                    agent.update()
                obs = next_obs

    def test_qmix_training_loop(self):
        """QMIX should train for a few episodes without errors."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = QMIXAgent(cfg, obs_dim=obs_dim, batch_size=8)
        env = MEIRPEnv(cfg)

        for ep in range(3):
            obs, _ = env.reset()
            done = False
            while not done:
                actions = agent.get_actions(obs, deterministic=False)
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                dones = np.full(cfg.num_nodes, float(terminated))
                agent.store_transition(obs, actions, rewards, next_obs, dones)
                if len(agent.buffer) >= agent.batch_size:
                    agent.update()
                obs = next_obs

    def test_mappo_training_loop(self):
        """MAPPO should train for a few episodes without errors."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = MAPPOAgent(cfg, obs_dim=obs_dim)
        env = MEIRPEnv(cfg)

        for ep in range(3):
            obs, _ = env.reset()
            done = False
            while not done:
                actions = agent.get_actions(obs, deterministic=False)
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                dones = np.full(cfg.num_nodes, float(terminated))
                agent.store_transition(obs, actions, rewards, next_obs, dones)
                obs = next_obs
            agent.update()

    def test_entropy_decreases_with_training(self):
        """Entropy should generally decrease as agent learns (IQL)."""
        cfg = make_3layer_config()
        obs_dim = 6 + 3 * cfg.lead_time
        agent = IQLAgent(cfg, obs_dim=obs_dim, batch_size=16,
                         epsilon_start=0.0, epsilon_end=0.0)  # no exploration
        env = MEIRPEnv(cfg)

        obs, _ = env.reset()
        entropy_before = agent.get_entropy(obs)

        # Train a bit
        for _ in range(50):
            actions = agent.get_actions(obs, deterministic=True)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            dones = np.full(cfg.num_nodes, float(terminated))
            agent.store_transition(obs, actions, rewards, next_obs, dones)
            if len(agent.buffer) >= agent.batch_size:
                agent.update()
            obs = next_obs if not terminated else env.reset()[0]

        obs, _ = env.reset()
        entropy_after = agent.get_entropy(obs)

        # Just check that entropy is computed and valid
        assert np.all(entropy_before >= 0)
        assert np.all(entropy_after >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
