"""Tests for GA/ALNS inventory metaheuristic solvers."""

import numpy as np

from meirp_project.baselines.routed import with_routing
from meirp_project.core.config import EnvConfig
from meirp_project.core.env import MEIRPEnv
from meirp_project.evaluation.runner import EvalRunner
from meirp_project.solvers.alns_inventory_solver import ALNSInventorySolver
from meirp_project.solvers.alns_irp_solver import ALNSIRPSolver
from meirp_project.solvers.ga_inventory_solver import GAInventorySolver


def make_cfg():
    base = EnvConfig(
        num_nodes=3,
        edges=[(0, 1), (0, 2)],
        layers=[0, 1, 1],
        H=[1.0, 2.0, 2.0],
        B=[5.0, 4.0, 4.0],
        max_order=12,
        lead_time=2,
        episode_len=5,
        demand_mean=5.0,
        init_inventory=0.0,
        init_pipeline=0.0,
        seed=11,
        coords=[(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)],
    )
    return with_routing(
        base,
        method="savings",
        use_2opt=True,
        transport_cost=0.5,
        vehicle_capacity=30.0,
        vehicle_count=2,
    )


def assert_valid_bdq(actions, cfg):
    assert len(actions) == cfg.num_nodes
    for action in actions:
        if isinstance(action, list):
            assert all(isinstance(x, int) for x in action)
            assert all(0 <= x <= cfg.max_order for x in action)
            assert sum(action) <= cfg.max_order
        else:
            assert isinstance(action, int)
            assert 0 <= action <= cfg.max_order


def test_ga_inventory_solver_valid_deterministic_and_basestock_seeded():
    cfg = make_cfg()
    env1 = MEIRPEnv(cfg)
    env2 = MEIRPEnv(cfg)
    env1.reset(seed=123)
    env2.reset(seed=123)

    solver1 = GAInventorySolver(cfg, population_size=8, n_generations=4, planning_horizon=3, seed=99)
    solver2 = GAInventorySolver(cfg, population_size=8, n_generations=4, planning_horizon=3, seed=99)
    actions1 = solver1.get_actions(env1)
    actions2 = solver2.get_actions(env2)

    assert actions1 == actions2
    assert_valid_bdq(actions1, cfg)
    assert np.isfinite(solver1.last_best_cost)
    assert solver1.last_best_cost <= solver1.last_basestock_cost + 1e-6


def test_alns_inventory_solver_valid_deterministic_and_basestock_seeded():
    cfg = make_cfg()
    env1 = MEIRPEnv(cfg)
    env2 = MEIRPEnv(cfg)
    env1.reset(seed=123)
    env2.reset(seed=123)

    solver1 = ALNSInventorySolver(cfg, max_iterations=8, planning_horizon=3, seed=99)
    solver2 = ALNSInventorySolver(cfg, max_iterations=8, planning_horizon=3, seed=99)
    actions1 = solver1.get_actions(env1)
    actions2 = solver2.get_actions(env2)

    assert actions1 == actions2
    assert_valid_bdq(actions1, cfg)
    assert np.isfinite(solver1.last_best_cost)
    assert solver1.last_best_cost <= solver1.last_basestock_cost + 1e-6


def test_metaheuristic_inventory_solvers_run_with_heuristic_routing_metrics():
    cfg = make_cfg()
    specs = [
        GAInventorySolver(cfg, population_size=8, n_generations=3, planning_horizon=3, seed=7),
        ALNSInventorySolver(cfg, max_iterations=8, planning_horizon=3, seed=7),
    ]
    for solver in specs:
        result = EvalRunner(cfg).eval_baseline(solver, n_episodes=1)
        for key in [
            "avg_cost_per_step",
            "avg_fill_rate",
            "total_transport",
            "avg_route_distance_per_step",
            "routing_feasible_rate",
        ]:
            assert key in result
            assert np.isfinite(result[key])
        assert result["total_transport"] > 0.0
        assert result["total_route_distance"] > 0.0
        assert result["routing_feasible_rate"] == 1.0


def test_alns_irp_solver_valid_route_aware_and_runs_with_metrics():
    cfg = make_cfg()
    env = MEIRPEnv(cfg)
    env.reset(seed=123)
    solver = ALNSIRPSolver(cfg, max_iterations=8, planning_horizon=3, seed=9)
    actions = solver.get_actions(env)

    assert_valid_bdq(actions, cfg)
    assert np.isfinite(solver.last_best_cost)
    assert solver.last_best_cost <= solver.last_basestock_cost + 1e-6

    result = EvalRunner(cfg).eval_baseline(
        ALNSIRPSolver(cfg, max_iterations=8, planning_horizon=3, seed=9),
        n_episodes=1,
    )
    for key in ["avg_cost_per_step", "total_transport", "total_route_distance", "routing_feasible_rate"]:
        assert key in result
        assert np.isfinite(result[key])
    assert result["total_transport"] > 0.0
    assert result["total_route_distance"] > 0.0
    assert result["routing_feasible_rate"] == 1.0
