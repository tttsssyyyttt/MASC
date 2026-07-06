"""Unit tests for core environment modules."""

import numpy as np
import pytest

from meirp_project.core.config import EnvConfig, from_yaml
from meirp_project.core.network import build_network
from meirp_project.core.demand import DemandGenerator
from meirp_project.core.allocation import allocate
from meirp_project.core.reward import compute_reward
from meirp_project.core.env import MEIRPEnv


# ============================================================
# Fixtures
# ============================================================

def make_2layer_cfg() -> EnvConfig:
    """Simple 1-2 two-layer network:
        0 (root)
       / \
      1   2 (terminals)
    """
    return EnvConfig(
        num_nodes=3,
        edges=[(0, 1), (0, 2)],
        layers=[0, 1, 1],
        H=[1.0, 2.0, 2.0],
        B=[5.0, 4.0, 4.0],
        max_order=20,
        lead_time=2,
        episode_len=30,
        demand_mean=8.0,
        seed=42,
    )


def make_3layer_cfg() -> EnvConfig:
    """1-2-4 three-layer network:
          0
         / \
        1   2
       /|   |\
      3  4  5  6  (terminals)
    """
    return EnvConfig(
        num_nodes=7,
        edges=[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        layers=[0, 1, 1, 2, 2, 2, 2],
        H=[1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
        B=[5.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0],
        max_order=20,
        lead_time=2,
        episode_len=30,
        demand_mean=8.0,
        seed=42,
    )


# ============================================================
# config.py tests
# ============================================================

class TestConfig:
    def test_valid_config(self):
        cfg = make_3layer_cfg()
        assert cfg.num_nodes == 7
        assert len(cfg.edges) == 6
        assert len(cfg.layers) == 7

    def test_invalid_layers_length(self):
        with pytest.raises(ValueError, match="layers length"):
            EnvConfig(
                num_nodes=3, edges=[(0, 1), (0, 2)],
                layers=[0, 1],  # wrong length
                H=[1, 2, 3], B=[1, 2, 3],
            )

    def test_invalid_H_length(self):
        with pytest.raises(ValueError, match="H length"):
            EnvConfig(
                num_nodes=3, edges=[(0, 1), (0, 2)],
                layers=[0, 1, 1],
                H=[1, 2], B=[1, 2, 3],  # H wrong length
            )

    def test_edge_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            EnvConfig(
                num_nodes=3, edges=[(0, 5)],  # 5 >= 3
                layers=[0, 1, 1],
                H=[1, 2, 3], B=[1, 2, 3],
            )

    def test_self_loop(self):
        with pytest.raises(ValueError, match="Self-loop"):
            EnvConfig(
                num_nodes=3, edges=[(1, 1)],
                layers=[0, 1, 1],
                H=[1, 2, 3], B=[1, 2, 3],
            )

    def test_duplicate_edge(self):
        with pytest.raises(ValueError, match="Duplicate"):
            EnvConfig(
                num_nodes=3, edges=[(0, 1), (0, 1)],
                layers=[0, 1, 1],
                H=[1, 2, 3], B=[1, 2, 3],
            )

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            EnvConfig(
                num_nodes=3, edges=[(0, 1), (0, 2)],
                layers=[0, 1, 1],
                H=[1, 2, 3], B=[1, 2, 3],
                alpha=1.5,
            )


# ============================================================
# network.py tests
# ============================================================

class TestNetwork:
    def test_2layer_topology(self):
        cfg = make_2layer_cfg()
        upstream, downstream, topo, terminals = build_network(cfg)

        # Node 0 is root: no upstream, 2 downstream
        assert upstream[0] == []
        assert sorted(downstream[0]) == [1, 2]

        # Node 1: upstream=[0], no downstream -> terminal
        assert upstream[1] == [0]
        assert downstream[1] == []
        assert 1 in terminals

        # Node 2: upstream=[0], no downstream -> terminal
        assert upstream[2] == [0]
        assert 2 in terminals

        assert terminals == [1, 2]

    def test_3layer_topology(self):
        cfg = make_3layer_cfg()
        upstream, downstream, topo, terminals = build_network(cfg)

        assert upstream[0] == []
        assert sorted(downstream[0]) == [1, 2]

        assert upstream[3] == [1]
        assert upstream[5] == [2]

        assert sorted(terminals) == [3, 4, 5, 6]

    def test_topological_order(self):
        cfg = make_3layer_cfg()
        _, _, topo, _ = build_network(cfg)

        # Every node appears exactly once
        assert sorted(topo) == list(range(7))

        # Node 0 must come before nodes 1,2; nodes 1,2 before 3,4,5,6
        pos = {node: idx for idx, node in enumerate(topo)}
        assert pos[0] < pos[1]
        assert pos[0] < pos[2]
        assert pos[1] < pos[3]
        assert pos[1] < pos[4]
        assert pos[2] < pos[5]
        assert pos[2] < pos[6]

    def test_multi_upstream(self):
        """Node 2 has two upstreams: 0 and 1."""
        cfg = EnvConfig(
            num_nodes=4,
            edges=[(0, 2), (1, 2), (2, 3)],
            layers=[0, 0, 1, 2],
            H=[1, 1, 2, 3], B=[5, 5, 4, 3],
        )
        upstream, downstream, _, terminals = build_network(cfg)

        assert sorted(upstream[2]) == [0, 1]
        assert downstream[2] == [3]
        assert terminals == [3]


# ============================================================
# demand.py tests
# ============================================================

class TestDemand:
    def test_poisson_shape(self):
        cfg = make_3layer_cfg()
        gen = DemandGenerator(cfg)
        demands = gen.generate(4)
        assert demands.shape == (4,)
        assert (demands >= 0).all()

    def test_deterministic_with_seed(self):
        cfg = make_3layer_cfg()
        gen1 = DemandGenerator(cfg)
        gen2 = DemandGenerator(cfg)
        d1 = gen1.generate(4)
        d2 = gen2.generate(4)
        np.testing.assert_array_equal(d1, d2)

    def test_normal_non_negative(self):
        cfg = EnvConfig(
            num_nodes=3, edges=[(0, 1), (0, 2)], layers=[0, 1, 1],
            H=[1, 2, 3], B=[5, 4, 3],
            demand_type="normal", demand_std=0.01, seed=42,
        )
        gen = DemandGenerator(cfg)
        for _ in range(100):
            d = gen.generate(4)
            assert (d >= 0).all()

    def test_merton(self):
        cfg = EnvConfig(
            num_nodes=3, edges=[(0, 1), (0, 2)], layers=[0, 1, 1],
            H=[1, 2, 3], B=[5, 4, 3],
            demand_type="merton", seed=42,
        )
        gen = DemandGenerator(cfg)
        d = gen.generate(4)
        assert d.shape == (4,)
        assert (d >= 0).all()


# ============================================================
# allocation.py tests
# ============================================================

class TestAllocation:
    def test_single_upstream_sufficient(self):
        shipments = allocate(10, [1.0], [100.0])
        assert shipments == [10.0]

    def test_single_upstream_insufficient(self):
        shipments = allocate(10, [1.0], [5.0])
        assert shipments == [5.0]

    def test_two_upstream_sufficient(self):
        shipments = allocate(10, [0.6, 0.4], [100.0, 100.0])
        assert abs(shipments[0] - 6.0) < 0.01
        assert abs(shipments[1] - 4.0) < 0.01
        assert abs(sum(shipments) - 10.0) < 0.01

    def test_two_upstream_one_short(self):
        # Upstream 0 has 100, upstream 1 only has 2
        # Requested: 6 + 4 = 10, but upstream 1 can only give 2
        # Remaining 2 should be redistributed to upstream 0
        shipments = allocate(10, [0.6, 0.4], [100.0, 2.0])
        assert abs(shipments[1] - 2.0) < 0.01
        # Upstream 0 should get its 6 + remaining 2 = 8
        assert abs(shipments[0] - 8.0) < 0.01

    def test_total_cannot_exceed_q(self):
        shipments = allocate(10, [0.5, 0.5], [3.0, 3.0])
        assert sum(shipments) <= 10.0 + 1e-6

    def test_zero_order(self):
        shipments = allocate(0, [1.0], [100.0])
        assert shipments == [0.0]


# ============================================================
# reward.py tests
# ============================================================

class TestReward:
    def test_basic_reward(self):
        inventory = np.array([10.0, 5.0])
        backlog = np.array([2.0, 0.0])
        demand = np.array([5.0, 3.0])
        H = np.array([1.0, 2.0])
        B = np.array([5.0, 4.0])

        rewards, hc, bc = compute_reward(inventory, backlog, demand, H, B, alpha=0.5)

        # holding_costs = [10, 10], backlog_costs = [10, 0]
        np.testing.assert_allclose(hc, [10.0, 10.0])
        np.testing.assert_allclose(bc, [10.0, 0.0])

        # r_local = [-20, -10], r_global = mean(r_local) = -15
        r_local = np.array([-20.0, -10.0])
        r_global = np.full(2, r_local.mean())
        expected = 0.5 * r_local + 0.5 * r_global
        np.testing.assert_allclose(rewards, expected)

    def test_alpha_zero(self):
        """alpha=0 -> all agents get the same global reward."""
        inventory = np.array([10.0, 0.0])
        backlog = np.array([0.0, 5.0])
        demand = np.array([2.0, 3.0])
        H = np.array([1.0, 1.0])
        B = np.array([1.0, 1.0])

        rewards, _, _ = compute_reward(inventory, backlog, demand, H, B, alpha=0.0)
        assert rewards[0] == rewards[1]  # both get global reward

    def test_alpha_one(self):
        """alpha=1 -> each agent gets only its own local reward."""
        inventory = np.array([10.0, 0.0])
        backlog = np.array([0.0, 5.0])
        demand = np.array([2.0, 3.0])
        H = np.array([1.0, 1.0])
        B = np.array([1.0, 1.0])

        rewards, _, _ = compute_reward(inventory, backlog, demand, H, B, alpha=1.0)
        assert rewards[0] == -10.0
        assert rewards[1] == -5.0


# ============================================================
# env.py tests
# ============================================================

class TestEnv:
    def test_reset(self):
        cfg = make_2layer_cfg()
        env = MEIRPEnv(cfg)
        obs, info = env.reset()

        assert obs.shape == (3, 6 + 3 * 2)  # N=3, obs_dim=6+3L=12
        assert env.inventory[0] == cfg.init_inventory
        assert env.backlog[0] == 0.0

    def test_step_returns_correct_shapes(self):
        cfg = make_2layer_cfg()
        env = MEIRPEnv(cfg)
        env.reset()

        # BDQ format: root -> int, non-root -> list[int]
        actions = [5, [5], [5]]

        obs, rewards, terminated, truncated, info = env.step(actions)

        assert obs.shape == (3, 12)
        assert rewards.shape == (3,)
        assert isinstance(terminated, bool)
        assert "holding_costs" in info
        assert "backlog_costs" in info
        assert "fill_rates" in info
        for key in [
            "inventory_demand",
            "downstream_order_demand",
            "shipped_demand",
            "external_demand",
            "fulfilled_demand",
            "inbound_shipment",
        ]:
            assert key in info
            assert info[key].shape == (3,)

    def test_pipeline_shift(self):
        """Pipeline should shift left and receive new shipment at the end."""
        cfg = make_2layer_cfg()
        env = MEIRPEnv(cfg)
        env.reset()

        # BDQ format
        actions = [10, [10], [10]]

        # Pipeline before step: all init_pipeline
        old_pipeline_last = env.pipeline[0, -1]

        obs, _, _, _, _ = env.step(actions)

        # After 1 step, pipeline[-1] should be shipment received for top node
        # Top node (0) has no upstream, so shipment_to[0] = q = 10
        # Terminal nodes (1,2) get shipment from node 0
        # Node 0: pipeline[-1] = 10 (external supplier)
        assert env.pipeline[0, -1] == 10.0

    def test_episode_terminates(self):
        cfg = EnvConfig(
            num_nodes=2, edges=[(0, 1)], layers=[0, 1],
            H=[1, 2], B=[5, 4],
            episode_len=3, seed=42,
        )
        env = MEIRPEnv(cfg)
        env.reset()

        actions = [5, [5]]
        for _ in range(3):
            obs, _, terminated, _, _ = env.step(actions)

        assert terminated

    def test_demand_propagation(self):
        """Non-terminal demand should equal downstream outgoing shipments to it."""
        cfg = make_2layer_cfg()
        env = MEIRPEnv(cfg)
        env.reset(seed=123)

        # BDQ format: root -> int, non-root -> list[int]
        actions = [0, [6], [4]]

        obs, _, _, _, info = env.step(actions)

        # Node 0's demand should be ~6 + 4 = 10 (if it has enough inventory)
        # Since node 0 starts with init_inventory=20, it can fulfill
        assert abs(env.demand[0] - 10.0) < 0.01
        assert abs(info["downstream_order_demand"][0] - 10.0) < 0.01
        assert abs(info["shipped_demand"][0] - 10.0) < 0.01
        assert abs(info["inventory_demand"][0] - env.demand[0]) < 0.01
        assert info["fulfilled_demand"][0] > 0.0

    def test_3layer_env(self):
        """3-layer env runs without errors for a full episode."""
        cfg = make_3layer_cfg()
        env = MEIRPEnv(cfg)
        env.reset()

        # BDQ format: node 0 root -> int, others non-root -> list[int]
        actions = [5, [5], [5], [5], [5], [5], [5]]

        for step in range(cfg.episode_len):
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert obs.shape == (7, 12)
            assert rewards.shape == (7,)
            if terminated:
                break

    def test_obs_normalization_range(self):
        """Observations should be in [-1, 1]."""
        cfg = make_3layer_cfg()
        env = MEIRPEnv(cfg)
        obs, _ = env.reset()

        assert (obs >= -1.0).all()
        assert (obs <= 1.0).all()

    def test_inventory_decreases_on_demand(self):
        """Terminal inventory should decrease when demand arrives."""
        cfg = EnvConfig(
            num_nodes=2, edges=[(0, 1)], layers=[0, 1],
            H=[1, 2], B=[5, 4],
            lead_time=1,  # short lead time for faster effect
            init_inventory=100.0,
            init_pipeline=0.0,
            demand_mean=20.0,
            seed=42,
        )
        env = MEIRPEnv(cfg)
        env.reset()

        # Don't order anything (BDQ format)
        actions = [0, [0]]

        initial_inv = env.inventory[1]
        for _ in range(5):
            env.step(actions)

        # Terminal inventory should have decreased due to demand
        assert env.inventory[1] < initial_inv or env.backlog[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
