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
    order_qty=None,
    fulfilled_demand=None,
) -> tuple:
    """Compute mixed reward for all nodes with optional stockout shaping.

    Args:
        inventory: shape (N,), current inventory
        backlog: shape (N,), current backlog
        demand: shape (N,), current step demand
        H: shape (N,), holding cost coefficients
        B: shape (N,), backlog cost coefficients
        alpha: mixing coefficient
        stockout_penalty: extra penalty per unit of unfulfilled demand (stockout)
        fill_rate_bonus: bonus per unit of fulfilled demand
        fulfilled_demand: actual fulfilled quantity from the environment

    Returns:
        rewards: shape (N,), mixed reward per node
        holding_costs: shape (N,), holding cost per node
        backlog_costs: shape (N,), backlog cost per node
    """
    holding_costs = H * inventory
    backlog_costs = B * backlog

    r_local = -(holding_costs + backlog_costs)

    if fulfilled_demand is None:
        # Original fallback inferred fulfillment from cumulative backlog:
        # fulfilled = np.maximum(0, demand - backlog)
        fulfilled = np.maximum(0, demand - backlog)
    else:
        fulfilled = np.minimum(np.maximum(fulfilled_demand, 0.0), demand)
    unfulfilled = np.maximum(0, demand - fulfilled)

    # Stockout shaping: extra penalty for current unfulfilled demand
    if stockout_penalty > 0:
        # Original version used cumulative backlog:
        # r_local = r_local - stockout_penalty * backlog
        r_local = r_local - stockout_penalty * unfulfilled

    # Fill rate bonus: reward for each unit of demand fulfilled
    if fill_rate_bonus > 0:
        r_local = r_local + fill_rate_bonus * fulfilled

    if order_penalty > 0 and order_qty is not None:
        orders = np.maximum(np.asarray(order_qty, dtype=np.float64), 0.0)
        r_local = r_local - order_penalty * orders

    # Use MEAN instead of SUM for global reward — avoids scaling with N agents
    r_global = np.full(len(inventory), r_local.mean())

    rewards = alpha * r_local + (1 - alpha) * r_global

    return rewards, holding_costs, backlog_costs
