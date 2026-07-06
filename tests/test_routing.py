"""Tests for routing helpers and routing-aware environment mode."""

import numpy as np
import pytest

from meirp_project.core.config import EnvConfig
from meirp_project.core.env import MEIRPEnv
from meirp_project.baselines.routed import RoutedBaseStockPolicy, with_routing
from meirp_project.evaluation.runner import EvalRunner
from meirp_project.routing.distance import route_distance
from meirp_project.routing.ga_routing import solve_ga_routing
from meirp_project.routing.nearest_insertion import solve_nearest_insertion
from meirp_project.routing.savings import solve_savings
from meirp_project.routing.two_opt import two_opt_route


def make_routing_cfg(routing_enabled=True, transport_cost=0.0):
    return EnvConfig(
        num_nodes=3,
        edges=[(0, 1), (0, 2)],
        layers=[0, 1, 1],
        H=[1.0, 2.0, 2.0],
        B=[5.0, 4.0, 4.0],
        max_order=20,
        lead_time=2,
        episode_len=5,
        demand_mean=8.0,
        seed=42,
        routing_enabled=routing_enabled,
        transport_cost=transport_cost,
        vehicle_capacity=30.0,
        vehicle_count=2,
        coords=[(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)],
    )


def test_route_distance_recomputes_known_triangle():
    coords = [(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)]
    assert abs(route_distance([0, 1, 2, 0], coords) - 12.0) <= 1e-6


def test_two_opt_never_increases_distance():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    route = [0, 2, 1, 3, 0]
    improved = two_opt_route(route, coords)
    assert route_distance(improved, coords) <= route_distance(route, coords) + 1e-9
    assert improved[0] == 0
    assert improved[-1] == 0


def test_nearest_insertion_visits_each_customer_once_and_respects_capacity():
    coords = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)]
    demands = {1: 1.0, 2: 1.0, 3: 1.0}
    plan = solve_nearest_insertion(
        depot=0,
        customers=[1, 2, 3],
        coords=coords,
        demands=demands,
        vehicle_capacity=3.0,
        vehicle_count=1,
    )
    visited = [node for route in plan.routes for node in route.stops]
    assert sorted(visited) == [1, 2, 3]
    assert all(route.load <= 3.0 + 1e-9 for route in plan.routes)
    assert plan.feasible


def test_savings_routes_start_end_at_depot_and_respect_capacity():
    coords = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)]
    demands = {1: 1.0, 2: 1.0, 3: 1.0}
    plan = solve_savings(
        depot=0,
        customers=[1, 2, 3],
        coords=coords,
        demands=demands,
        vehicle_capacity=2.0,
        vehicle_count=2,
    )
    visited = [node for route in plan.routes for node in route.stops]
    assert sorted(visited) == [1, 2, 3]
    assert all(route.node_sequence[0] == 0 and route.node_sequence[-1] == 0 for route in plan.routes)
    assert all(route.load <= 2.0 + 1e-9 for route in plan.routes)
    assert plan.feasible


def test_ga_routing_visits_each_customer_once_and_beats_simple_random_route():
    coords = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)]
    demands = {1: 1.0, 2: 1.0, 3: 1.0}
    plan = solve_ga_routing(
        depot=0,
        customers=[1, 2, 3],
        coords=coords,
        demands=demands,
        vehicle_capacity=3.0,
        vehicle_count=1,
        population_size=10,
        generations=8,
        seed=5,
    )
    visited = [node for route in plan.routes for node in route.stops]
    random_distance = route_distance([0, 3, 1, 2, 0], coords)
    assert sorted(visited) == [1, 2, 3]
    assert all(route.load <= 3.0 + 1e-9 for route in plan.routes)
    assert plan.feasible
    assert plan.total_distance <= random_distance + 1e-6


def test_routing_zero_cost_preserves_inventory_and_rewards():
    demand_sequence = np.array([[5.0, 6.0], [4.0, 3.0]], dtype=np.float64)
    actions = [0, [6], [4]]

    cfg_plain = make_routing_cfg(routing_enabled=False, transport_cost=0.0)
    cfg_routed = make_routing_cfg(routing_enabled=True, transport_cost=0.0)

    env_plain = MEIRPEnv(cfg_plain)
    env_routed = MEIRPEnv(cfg_routed)
    env_plain.reset(seed=123, options={"demand_sequence": demand_sequence})
    env_routed.reset(seed=123, options={"demand_sequence": demand_sequence})

    _, rewards_plain, _, _, info_plain = env_plain.step(actions)
    _, rewards_routed, _, _, info_routed = env_routed.step(actions)

    np.testing.assert_allclose(env_plain.inventory, env_routed.inventory, atol=1e-9)
    np.testing.assert_allclose(env_plain.backlog, env_routed.backlog, atol=1e-9)
    np.testing.assert_allclose(env_plain.pipeline, env_routed.pipeline, atol=1e-9)
    np.testing.assert_allclose(rewards_plain, rewards_routed, atol=1e-9)
    np.testing.assert_allclose(info_plain["holding_costs"], info_routed["holding_costs"], atol=1e-9)
    np.testing.assert_allclose(info_plain["backlog_costs"], info_routed["backlog_costs"], atol=1e-9)
    assert info_routed["route_distance"] > 0.0
    assert info_routed["total_transport_cost"] == pytest.approx(0.0)
    assert info_routed["routing_feasible"]


def test_positive_transport_cost_is_reported_and_penalizes_reward():
    demand_sequence = np.array([[5.0, 6.0]], dtype=np.float64)
    actions = [0, [6], [4]]

    cfg_zero = make_routing_cfg(routing_enabled=True, transport_cost=0.0)
    cfg_cost = make_routing_cfg(routing_enabled=True, transport_cost=0.5)

    env_zero = MEIRPEnv(cfg_zero)
    env_cost = MEIRPEnv(cfg_cost)
    env_zero.reset(seed=123, options={"demand_sequence": demand_sequence})
    env_cost.reset(seed=123, options={"demand_sequence": demand_sequence})

    _, rewards_zero, _, _, _ = env_zero.step(actions)
    _, rewards_cost, _, _, info_cost = env_cost.step(actions)

    assert info_cost["route_distance"] > 0.0
    assert info_cost["total_transport_cost"] > 0.0
    assert rewards_cost.sum() < rewards_zero.sum()


def test_routed_baseline_metrics_preserve_inventory_fields_and_add_route_fields():
    cfg = with_routing(
        make_routing_cfg(routing_enabled=False),
        method="savings",
        use_2opt=True,
        transport_cost=0.5,
        vehicle_capacity=30.0,
        vehicle_count=2,
    )
    runner = EvalRunner(cfg)
    result = runner.eval_baseline(RoutedBaseStockPolicy(cfg, "savings"), n_episodes=1)

    for key in ["total_holding", "total_backlog", "avg_fill_rate", "avg_cost_per_step"]:
        assert key in result
        assert np.isfinite(result[key])

    for key in ["total_transport", "avg_transport_per_step", "total_route_distance", "avg_route_distance_per_step"]:
        assert key in result
        assert np.isfinite(result[key])

    assert result["total_transport"] > 0.0
    assert result["total_route_distance"] > 0.0
    assert 0.0 <= result["routing_feasible_rate"] <= 1.0
