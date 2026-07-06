"""Reward computation — pure function, zero state."""

import numpy as np

from .config import EnvConfig


def compute_reward(
    inventory: np.ndarray,
    backlog: np.ndarray,
    demand: np.ndarray,
    H: np.ndarray,
    B: np.ndarray,
    alpha: float,
    stockout_penalty: float = 0.0,
    fill_rate_bonus: float = 0.0,
    order_penalty: float = 0.0,
    service_level_target: float = 0.0,
    service_level_penalty: float = 0.0,
    order_qty=None,
    fulfilled_demand=None,
) -> tuple:
    holding_costs = H * inventory
    backlog_costs = B * backlog

    # Base objective: minimize operational inventory cost.
    r_local = -(holding_costs + backlog_costs)

    if fulfilled_demand is None:
        fulfilled = np.maximum(0, demand - backlog)
    else:
        fulfilled = np.minimum(np.maximum(fulfilled_demand, 0.0), demand)

    unfulfilled = np.maximum(0, demand - fulfilled)

    # Optional old shaping. For the formal cost-only and service-level experiments,
    # set these to 0.0 in the config.
    if stockout_penalty > 0:
        r_local = r_local - stockout_penalty * unfulfilled

    if fill_rate_bonus > 0:
        r_local = r_local + fill_rate_bonus * fulfilled

    # Optional diagnostic regularization only. Do not use in formal experiments.
    if order_penalty > 0 and order_qty is not None:
        orders = np.maximum(np.asarray(order_qty, dtype=np.float64), 0.0)
        r_local = r_local - order_penalty * orders

    # Service-level soft constraint:
    # penalize fill rate below target, but give no extra reward above target.
    if service_level_penalty > 0 and service_level_target > 0:
        demand_arr = np.maximum(np.asarray(demand, dtype=np.float64), 0.0)
        fulfilled_arr = np.maximum(np.asarray(fulfilled, dtype=np.float64), 0.0)

        fill_rate = np.ones_like(demand_arr, dtype=np.float64)
        mask = demand_arr > 1e-9
        fill_rate[mask] = fulfilled_arr[mask] / demand_arr[mask]
        fill_rate = np.clip(fill_rate, 0.0, 1.0)

        service_gap = np.maximum(0.0, service_level_target - fill_rate)
        r_local = r_local - service_level_penalty * service_gap ** 2

    r_global = np.full(len(inventory), r_local.mean())
    rewards = alpha * r_local + (1 - alpha) * r_global

    return rewards, holding_costs, backlog_costs