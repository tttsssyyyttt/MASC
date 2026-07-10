# tests/test_env_demand_backlog.py

import numpy as np
import pytest

from core.config import EnvConfig
from core.env import MEIRPEnv


def make_two_node_cfg(
    *,
    lead_time: int = 1,
    max_order: int = 50,
    episode_len: int = 1,
    init_inventory: float = 0.0,
    init_pipeline: float = 0.0,
) -> EnvConfig:
    """Minimal deterministic 2-node chain: root 0 -> terminal 1."""
    return EnvConfig(
        num_nodes=2,
        edges=[(0, 1)],
        layers=[0, 1],
        H=[1.0, 1.0],
        B=[10.0, 10.0],
        max_order=max_order,
        lead_time=lead_time,
        episode_len=episode_len,
        demand_type="poisson",
        demand_mean=0.0,
        demand_std=0.0,
        alpha=1.0,
        reward_scale=1.0,
        stockout_penalty=0.0,
        fill_rate_bonus=0.0,
        order_penalty=0.0,
        service_level_target=0.0,
        service_level_penalty=0.0,
        init_inventory=init_inventory,
        init_pipeline=init_pipeline,
        routing_enabled=False,
        transport_cost=0.0,
        seed=123,
    )


def reset_with_terminal_demands(env: MEIRPEnv, terminal_demands):
    """Reset env with a fixed terminal demand sequence.

    terminal_demands should be shape (T, n_terminals).
    For this 2-node test network, n_terminals = 1.
    """
    demand_sequence = np.asarray(terminal_demands, dtype=np.float64)
    obs, info = env.reset(seed=0, options={"demand_sequence": demand_sequence})
    return obs, info


def assert_close(actual, expected, *, name: str, atol: float = 1e-9):
    assert np.isclose(actual, expected, atol=atol), (
        f"{name}: expected {expected}, got {actual}"
    )


def test_unmet_downstream_request_enters_upstream_backlog():
    """If child requests 20 but root has only 5, root backlog must become 15.

    This is the core anti-regression test for the old bug:
    demand[root] must be requested demand, not shipped demand.
    """
    cfg = make_two_node_cfg()
    env = MEIRPEnv(cfg)
    reset_with_terminal_demands(env, [[0.0]])

    # Root has only 5 available units.
    env.inventory[:] = 0.0
    env.pipeline[:] = 0.0
    env.backlog[:] = 0.0
    env.inventory[0] = 5.0

    # Actions:
    # node 0 is root: order 0 from external source.
    # node 1 is terminal with upstream [0]: request 20 from root.
    actions = [0, [20]]

    _, _, _, _, info = env.step(actions)

    assert_close(info["demand"][0], 20.0, name="root demand")
    assert_close(info["fulfilled_demand"][0], 5.0, name="root fulfilled_demand")
    assert_close(info["backlog"][0], 15.0, name="root backlog")
    assert_close(info["shipped_demand"][0], 5.0, name="root shipped_demand")
    assert_close(info["inbound_shipment"][1], 5.0, name="terminal inbound shipment")

    # Terminal has no external demand in this test.
    assert_close(info["demand"][1], 0.0, name="terminal demand")
    assert_close(info["backlog"][1], 0.0, name="terminal backlog")


def test_pipeline_arrival_is_available_before_shipment_allocation():
    """Pipeline arrival must be received before deciding feasible shipment.

    Regression target:
    old inventory = 0
    pipeline arrival = 30
    child requests 20
    expected root ships 20 and backlog stays 0.

    If allocation happens before receiving pipeline, shipped_demand[root] will
    incorrectly be 0.
    """
    cfg = make_two_node_cfg()
    env = MEIRPEnv(cfg)
    reset_with_terminal_demands(env, [[0.0]])

    env.inventory[:] = 0.0
    env.pipeline[:] = 0.0
    env.backlog[:] = 0.0

    # With lead_time=1, pipeline[:, 0] arrives at the start of this step.
    env.pipeline[0, 0] = 30.0

    actions = [0, [20]]

    _, _, _, _, info = env.step(actions)

    assert_close(info["demand"][0], 20.0, name="root demand")
    assert_close(info["fulfilled_demand"][0], 20.0, name="root fulfilled_demand")
    assert_close(info["backlog"][0], 0.0, name="root backlog")
    assert_close(info["shipped_demand"][0], 20.0, name="root shipped_demand")
    assert_close(info["inbound_shipment"][1], 20.0, name="terminal inbound shipment")

    # Root had 30 arrival, fulfilled 20, ordered 0 new units.
    # After step, remaining on-hand inventory should be 10.
    assert_close(info["inventory"][0], 10.0, name="root ending inventory")


def test_info_fields_use_consistent_demand_fulfillment_backlog_semantics():
    """Canonical and legacy info fields must not disagree after the refactor."""
    cfg = make_two_node_cfg()
    env = MEIRPEnv(cfg)
    reset_with_terminal_demands(env, [[7.0]])

    env.inventory[:] = 0.0
    env.pipeline[:] = 0.0
    env.backlog[:] = 0.0

    # Root can satisfy child replenishment request.
    env.inventory[0] = 12.0

    # Terminal can satisfy only part of external demand.
    env.inventory[1] = 3.0

    # node 1 requests 10 from root; external customer demand at terminal is 7.
    actions = [0, [10]]

    _, _, _, _, info = env.step(actions)

    demand = np.asarray(info["demand"], dtype=np.float64)
    inventory_demand = np.asarray(info["inventory_demand"], dtype=np.float64)
    fulfilled = np.asarray(info["fulfilled_demand"], dtype=np.float64)
    shipped = np.asarray(info["shipped_demand"], dtype=np.float64)
    backlog = np.asarray(info["backlog"], dtype=np.float64)
    fill_rates = np.asarray(info["fill_rates"], dtype=np.float64)

    # Canonical demand and legacy inventory_demand must be the same after refactor.
    np.testing.assert_allclose(inventory_demand, demand, atol=1e-9)

    # Node 0 serves downstream replenishment request = 10.
    assert_close(demand[0], 10.0, name="root demand")
    assert_close(fulfilled[0], 10.0, name="root fulfilled_demand")
    assert_close(backlog[0], 0.0, name="root backlog")
    assert_close(shipped[0], 10.0, name="root shipped_demand")

    # Node 1 serves external customer demand = 7.
    assert_close(demand[1], 7.0, name="terminal demand")
    assert_close(fulfilled[1], 3.0, name="terminal fulfilled_demand")
    assert_close(backlog[1], 4.0, name="terminal backlog")
    assert_close(shipped[1], 3.0, name="terminal shipped_demand")

    # Physical/logical sanity constraints.
    assert np.all(fulfilled <= demand + 1e-9), (
        f"fulfilled_demand must be <= demand, got fulfilled={fulfilled}, demand={demand}"
    )
    assert np.all(shipped <= demand + 1e-9), (
        f"shipped_demand must be <= demand, got shipped={shipped}, demand={demand}"
    )
    assert np.all(backlog >= -1e-9), f"backlog must be nonnegative, got {backlog}"

    # Current-period fill rate must be fulfilled / demand.
    np.testing.assert_allclose(
        fill_rates,
        np.array([1.0, 3.0 / 7.0]),
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "root_inventory, child_order, expected_fulfilled, expected_backlog",
    [
        (0.0, 20, 0.0, 20.0),
        (5.0, 20, 5.0, 15.0),
        (20.0, 20, 20.0, 0.0),
        (30.0, 20, 20.0, 0.0),
    ],
)
def test_root_backlog_boundary_cases(
    root_inventory,
    child_order,
    expected_fulfilled,
    expected_backlog,
):
    """Small boundary table for the same root-demand invariant."""
    cfg = make_two_node_cfg()
    env = MEIRPEnv(cfg)
    reset_with_terminal_demands(env, [[0.0]])

    env.inventory[:] = 0.0
    env.pipeline[:] = 0.0
    env.backlog[:] = 0.0
    env.inventory[0] = root_inventory

    actions = [0, [child_order]]

    _, _, _, _, info = env.step(actions)

    assert_close(info["demand"][0], float(child_order), name="root demand")
    assert_close(
        info["fulfilled_demand"][0],
        expected_fulfilled,
        name="root fulfilled_demand",
    )
    assert_close(info["backlog"][0], expected_backlog, name="root backlog")
    assert_close(
        info["shipped_demand"][0],
        expected_fulfilled,
        name="root shipped_demand",
    )