"""Tests for OR-Tools rolling-horizon IRP solver."""

import numpy as np
import pytest

pytest.importorskip("ortools.sat.python.cp_model")

from meirp_project.baselines.routed import with_routing
from meirp_project.core.config import EnvConfig
from meirp_project.core.env import MEIRPEnv
from meirp_project.evaluation.runner import EvalRunner
from meirp_project.solvers.ortools_irp_solver import ORToolsIRPSolver


def make_cfg():
    base = EnvConfig(
        num_nodes=3,
        edges=[(0, 1), (0, 2)],
        layers=[0, 1, 1],
        H=[1.0, 2.0, 2.0],
        B=[5.0, 4.0, 4.0],
        max_order=12,
        lead_time=2,
        episode_len=4,
        demand_mean=5.0,
        init_inventory=0.0,
        init_pipeline=0.0,
        seed=23,
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


def test_ortools_solver_returns_valid_action_and_beats_basestock_internal_objective():
    cfg = make_cfg()
    env = MEIRPEnv(cfg)
    env.reset(seed=123)
    solver = ORToolsIRPSolver(cfg, planning_horizon=3, time_limit=3.0)
    actions = solver.get_actions(env)

    assert_valid_bdq(actions, cfg)
    assert solver.last_status in {"OPTIMAL", "FEASIBLE"}
    assert np.isfinite(solver.last_objective)
    assert np.isfinite(solver.last_basestock_objective)
    assert solver.last_objective <= solver.last_basestock_objective + 1e-6


def test_ortools_solver_runs_with_routing_metrics():
    cfg = make_cfg()
    solver = ORToolsIRPSolver(cfg, planning_horizon=3, time_limit=3.0)
    result = EvalRunner(cfg).eval_baseline(solver, n_episodes=1)

    for key in [
        "avg_cost_per_step",
        "avg_fill_rate",
        "total_transport",
        "total_route_distance",
        "routing_feasible_rate",
    ]:
        assert key in result
        assert np.isfinite(result[key])
    assert result["total_transport"] >= 0.0
    assert result["total_route_distance"] >= 0.0
    assert 0.0 <= result["routing_feasible_rate"] <= 1.0
