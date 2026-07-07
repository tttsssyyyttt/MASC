"""Metrics computation for supply chain evaluation.

Includes: cost breakdown, fill rates, bullwhip effect (CV), inventory turnover.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def compute_metrics(info_history: List[dict], cfg=None) -> dict:
    """Compute evaluation metrics from a history of step info dicts.

    Args:
        info_history: list of info dicts from env.step()
        cfg: optional EnvConfig for additional metrics

    Returns:
        metrics: dict of metric values
    """
    if not info_history:
        return {}

    # Aggregate costs
    total_holding = sum(info["holding_costs"].sum() for info in info_history)
    total_backlog = sum(info["backlog_costs"].sum() for info in info_history)
    total_transport = sum(float(info.get("total_transport_cost", 0.0)) for info in info_history)
    total_inventory_cost = total_holding + total_backlog
    weighted_total_cost = total_inventory_cost + total_transport
    total_cost = weighted_total_cost
    total_route_distance = sum(float(info.get("route_distance", 0.0)) for info in info_history)
    routing_feasible_rate = np.mean([1.0 if info.get("routing_feasible", True) else 0.0 for info in info_history])
    transport_weight = float(getattr(cfg, "transport_cost", 0.0)) if cfg is not None else 0.0

    n_steps = len(info_history)
    n_agents = info_history[0]["holding_costs"].shape[0]

    # Fill rates
    # fill_rates = np.array([info["fill_rates"] for info in info_history])  # (T, N)
    # avg_fill_rate = fill_rates.mean()
    #
    # min_fill_rate = fill_rates.min()
    # per_agent_fill_rate = fill_rates.mean(axis=0)  # (N,)
    #
    # orders = np.array([info["order_qty"] for info in info_history])  # (T, N)
    # demands = np.array([info["demand"] for info in info_history])  # legacy inventory demand
    # requested_demands = np.array([
    #     info.get("downstream_order_demand", info["demand"]) for info in info_history
    # ])
    # shipped_demands = np.array([
    #     info.get("shipped_demand", info["demand"]) for info in info_history
    # ])
    # external_demands = np.array([
    #     info.get("external_demand", np.zeros(n_agents)) for info in info_history
    # ])
    # fulfilled_demands = np.array([
    #     info.get("fulfilled_demand", info["demand"]) for info in info_history
    # ])
    # Fill rates
    fill_rates = np.array([info["fill_rates"] for info in info_history])  # (T, N)

    orders = np.array([info["order_qty"] for info in info_history])  # (T, N)
    demands = np.array([info["demand"] for info in info_history])  # legacy inventory demand
    requested_demands = np.array([
        info.get("downstream_order_demand", info["demand"]) for info in info_history
    ])
    shipped_demands = np.array([
        info.get("shipped_demand", info["demand"]) for info in info_history
    ])
    external_demands = np.array([
        info.get("external_demand", np.zeros(n_agents)) for info in info_history
    ])
    fulfilled_demands = np.array([
        info.get("fulfilled_demand", info["demand"]) for info in info_history
    ])

    # MEIRP demand-weighted installation fill rate:
    # - terminal layer serves external customer demand
    # - upstream/intermediate layers serve downstream replenishment demand
    service_demands = requested_demands.copy()
    if cfg is not None and hasattr(cfg, "layers"):
        layers = np.asarray(cfg.layers)
        terminal_mask = layers == layers.max()
        service_demands[:, terminal_mask] = external_demands[:, terminal_mask]

    fulfilled_for_service = np.minimum(fulfilled_demands, service_demands)

    total_service_demand = service_demands.sum()
    total_fulfilled = fulfilled_for_service.sum()
    avg_fill_rate = float(total_fulfilled / max(total_service_demand, 1e-9))

    per_node_service_demand = service_demands.sum(axis=0)
    per_node_fulfilled = fulfilled_for_service.sum(axis=0)
    per_agent_fill_rate = np.divide(
        per_node_fulfilled,
        np.maximum(per_node_service_demand, 1e-9),
    )

    min_fill_rate = float(per_agent_fill_rate.min())

    per_node_order_mean = orders.mean(axis=0)
    per_node_order_std = orders.std(axis=0)
    per_node_order_var = orders.var(axis=0)
    per_node_demand_mean = demands.mean(axis=0)
    per_node_demand_std = demands.std(axis=0)
    per_node_demand_var = demands.var(axis=0)
    per_node_requested_demand_mean = requested_demands.mean(axis=0)
    per_node_requested_demand_std = requested_demands.std(axis=0)
    per_node_requested_demand_var = requested_demands.var(axis=0)
    per_node_shipped_demand_mean = shipped_demands.mean(axis=0)
    per_node_shipped_demand_std = shipped_demands.std(axis=0)
    per_node_shipped_demand_var = shipped_demands.var(axis=0)
    per_node_external_demand_mean = external_demands.mean(axis=0)
    per_node_external_demand_std = external_demands.std(axis=0)
    per_node_external_demand_var = external_demands.var(axis=0)
    per_node_fulfilled_demand_mean = fulfilled_demands.mean(axis=0)
    per_node_fulfilled_demand_std = fulfilled_demands.std(axis=0)
    per_node_fulfilled_demand_var = fulfilled_demands.var(axis=0)

    # Bullwhip effect: variance of orders / variance of demand per node.
    bullwhip_cv = _compute_bullwhip_cv(info_history)
    requested_bullwhip_cv = _compute_bullwhip_cv(info_history, "downstream_order_demand")
    shipped_bullwhip_cv = _compute_bullwhip_cv(info_history, "shipped_demand")

    # Inventory turnover: total_demand / avg_inventory per node
    inventory_turnover = _compute_inventory_turnover(info_history)

    # Stockout rate = 1 - fill rate
    avg_stockout_rate = 1.0 - avg_fill_rate
    min_stockout_rate = 1.0 - min_fill_rate
    per_agent_stockout_rate = 1.0 - per_agent_fill_rate  # (N,)

    # Per-node costs
    per_node_holding = np.array([info["holding_costs"] for info in info_history]).sum(axis=0)  # (N,)
    per_node_backlog = np.array([info["backlog_costs"] for info in info_history]).sum(axis=0)  # (N,)
    per_node_transport = np.array([
        info.get("transport_costs", np.zeros(n_agents)) for info in info_history
    ]).sum(axis=0)  # (N,)
    per_node_cost = per_node_holding + per_node_backlog + per_node_transport  # (N,)
    per_node_cost_per_step = per_node_cost / n_steps  # (N,)

    # Per-node average inventory and backlog levels
    per_node_avg_inventory = np.array([info["inventory"] for info in info_history]).mean(axis=0)  # (N,)
    per_node_avg_backlog = np.array(
        [info.get("backlog_costs", np.zeros(n_agents)) / np.maximum(cfg.B, 1e-10) if cfg is not None else np.zeros(n_agents)
         for info in info_history]
    ).mean(axis=0) if cfg is not None else np.zeros(n_agents)

    # Safe bullwhip mean (ignoring inf)
    finite_bw = np.where(np.isfinite(bullwhip_cv), bullwhip_cv, np.nan)
    avg_bullwhip_cv = float(np.nanmean(finite_bw)) if not np.all(np.isnan(finite_bw)) else 1.0
    finite_requested_bw = np.where(np.isfinite(requested_bullwhip_cv), requested_bullwhip_cv, np.nan)
    avg_requested_bullwhip_cv = (
        float(np.nanmean(finite_requested_bw)) if not np.all(np.isnan(finite_requested_bw)) else 1.0
    )
    finite_shipped_bw = np.where(np.isfinite(shipped_bullwhip_cv), shipped_bullwhip_cv, np.nan)
    avg_shipped_bullwhip_cv = (
        float(np.nanmean(finite_shipped_bw)) if not np.all(np.isnan(finite_shipped_bw)) else 1.0
    )

    metrics = {
        # Global aggregates
        "total_cost": total_cost,
        "avg_cost_per_step": total_cost / n_steps,
        "weighted_total_cost": weighted_total_cost,
        "avg_weighted_total_cost_per_step": weighted_total_cost / n_steps,
        "total_inventory_cost": total_inventory_cost,
        "avg_inventory_cost_per_step": total_inventory_cost / n_steps,
        "total_holding": total_holding,
        "total_backlog": total_backlog,
        "total_transport": total_transport,
        "routing_cost": total_transport,
        "avg_routing_cost_per_step": total_transport / n_steps,
        "transport_weight": transport_weight,
        "avg_transport_per_step": total_transport / n_steps,
        "total_route_distance": total_route_distance,
        "avg_route_distance_per_step": total_route_distance / n_steps,
        "routing_feasible_rate": float(routing_feasible_rate),
        "holding_ratio": total_holding / (total_cost + 1e-10),
        "backlog_ratio": total_backlog / (total_cost + 1e-10),
        "transport_ratio": total_transport / (total_cost + 1e-10),
        "avg_fill_rate": avg_fill_rate,
        "min_fill_rate": min_fill_rate,
        "avg_stockout_rate": avg_stockout_rate,
        "min_stockout_rate": min_stockout_rate,
        # Per-node arrays (N,)
        "per_node_cost": per_node_cost,
        "per_node_cost_per_step": per_node_cost_per_step,
        "per_node_holding": per_node_holding,
        "per_node_backlog": per_node_backlog,
        "per_node_transport": per_node_transport,
        "per_node_fill_rate": per_agent_fill_rate,
        "per_node_stockout_rate": per_agent_stockout_rate,
        "per_node_bullwhip_cv": bullwhip_cv,
        "per_node_order_mean": per_node_order_mean,
        "per_node_order_std": per_node_order_std,
        "per_node_order_var": per_node_order_var,
        "per_node_demand_mean": per_node_demand_mean,
        "per_node_demand_std": per_node_demand_std,
        "per_node_demand_var": per_node_demand_var,
        "per_node_requested_demand_mean": per_node_requested_demand_mean,
        "per_node_requested_demand_std": per_node_requested_demand_std,
        "per_node_requested_demand_var": per_node_requested_demand_var,
        "per_node_shipped_demand_mean": per_node_shipped_demand_mean,
        "per_node_shipped_demand_std": per_node_shipped_demand_std,
        "per_node_shipped_demand_var": per_node_shipped_demand_var,
        "per_node_external_demand_mean": per_node_external_demand_mean,
        "per_node_external_demand_std": per_node_external_demand_std,
        "per_node_external_demand_var": per_node_external_demand_var,
        "per_node_fulfilled_demand_mean": per_node_fulfilled_demand_mean,
        "per_node_fulfilled_demand_std": per_node_fulfilled_demand_std,
        "per_node_fulfilled_demand_var": per_node_fulfilled_demand_var,
        "per_node_turnover": inventory_turnover,
        "per_node_avg_inventory": per_node_avg_inventory,
        # Global means (for backward compatibility)
        "per_agent_fill_rate": per_agent_fill_rate,
        "bullwhip_cv": bullwhip_cv,
        "avg_bullwhip_cv": avg_bullwhip_cv,
        "requested_bullwhip_cv": requested_bullwhip_cv,
        "avg_requested_bullwhip_cv": avg_requested_bullwhip_cv,
        "shipped_bullwhip_cv": shipped_bullwhip_cv,
        "avg_shipped_bullwhip_cv": avg_shipped_bullwhip_cv,
        "avg_order_mean": float(per_node_order_mean.mean()),
        "avg_order_std": float(per_node_order_std.mean()),
        "avg_order_var": float(per_node_order_var.mean()),
        "avg_demand_mean": float(per_node_demand_mean.mean()),
        "avg_demand_std": float(per_node_demand_std.mean()),
        "avg_demand_var": float(per_node_demand_var.mean()),
        "avg_requested_demand_mean": float(per_node_requested_demand_mean.mean()),
        "avg_requested_demand_std": float(per_node_requested_demand_std.mean()),
        "avg_requested_demand_var": float(per_node_requested_demand_var.mean()),
        "avg_shipped_demand_mean": float(per_node_shipped_demand_mean.mean()),
        "avg_shipped_demand_std": float(per_node_shipped_demand_std.mean()),
        "avg_shipped_demand_var": float(per_node_shipped_demand_var.mean()),
        "avg_external_demand_mean": float(per_node_external_demand_mean.mean()),
        "avg_external_demand_std": float(per_node_external_demand_std.mean()),
        "avg_external_demand_var": float(per_node_external_demand_var.mean()),
        "avg_fulfilled_demand_mean": float(per_node_fulfilled_demand_mean.mean()),
        "avg_fulfilled_demand_std": float(per_node_fulfilled_demand_std.mean()),
        "avg_fulfilled_demand_var": float(per_node_fulfilled_demand_var.mean()),
        "min_demand_var": float(per_node_demand_var.min()),
        "min_requested_demand_var": float(per_node_requested_demand_var.min()),
        "max_order_var": float(per_node_order_var.max()),
        "max_bullwhip_cv": float(np.nanmax(finite_bw)) if not np.all(np.isnan(finite_bw)) else 1.0,
        "max_requested_bullwhip_cv": (
            float(np.nanmax(finite_requested_bw)) if not np.all(np.isnan(finite_requested_bw)) else 1.0
        ),
        "inventory_turnover": inventory_turnover,
        "avg_inventory_turnover": inventory_turnover.mean(),
        "n_steps": n_steps,
    }

    # Per-layer aggregation if cfg is provided
    if cfg is not None:
        layers = np.array(cfg.layers)
        unique_layers = sorted(set(layers))
        per_layer = {}
        for l in unique_layers:
            mask = layers == l
            per_layer[f"layer_{l}_cost_per_step"] = float(per_node_cost_per_step[mask].mean())
            per_layer[f"layer_{l}_fill_rate"] = float(per_agent_fill_rate[mask].mean())
            per_layer[f"layer_{l}_stockout_rate"] = float(per_agent_stockout_rate[mask].mean())
            layer_bw = bullwhip_cv[mask]
            layer_bw_finite = layer_bw[np.isfinite(layer_bw)]
            per_layer[f"layer_{l}_bullwhip_cv"] = float(layer_bw_finite.mean()) if len(layer_bw_finite) > 0 else 1.0
            per_layer[f"layer_{l}_turnover"] = float(inventory_turnover[mask].mean())
            per_layer[f"layer_{l}_holding"] = float(per_node_holding[mask].mean())
            per_layer[f"layer_{l}_backlog"] = float(per_node_backlog[mask].mean())
            per_layer[f"layer_{l}_transport"] = float(per_node_transport[mask].mean())
        metrics["per_layer"] = per_layer
        metrics["n_layers"] = len(unique_layers)

    return metrics


def _compute_bullwhip_cv(info_history: List[dict], demand_key: str = "demand") -> np.ndarray:
    """Compute bullwhip effect per node.

    Uses the standard supply chain definition:
    bullwhip = Var(orders) / Var(demand)

    A ratio > 1 indicates demand variability amplification (bullwhip effect).
    A ratio < 1 indicates demand smoothing (good).
    A ratio of 0 means no ordering variability (agent not responding to demand).

    Returns:
        bullwhip_cv: shape (N,), variance ratio of orders / demand per node
    """
    n_agents = info_history[0]["order_qty"].shape[0]

    # Collect order and demand series
    orders = np.array([info["order_qty"] for info in info_history])  # (T, N)
    demands = np.array([info.get(demand_key, info["demand"]) for info in info_history])  # (T, N)

    bullwhip = np.zeros(n_agents)

    for i in range(n_agents):
        var_order = orders[:, i].var()
        var_demand = demands[:, i].var()

        if var_demand > 1e-10:
            bullwhip[i] = var_order / var_demand
        elif var_order > 1e-10:
            bullwhip[i] = np.inf
        else:
            bullwhip[i] = 1.0

    return bullwhip


def _compute_inventory_turnover(info_history: List[dict]) -> np.ndarray:
    """Compute inventory turnover rate per node.

    Turnover = total_demand_served / avg_inventory
    Higher turnover = more efficient inventory management.

    Returns:
        turnover: shape (N,), inventory turnover per node
    """
    n_agents = info_history[0]["inventory"].shape[0]

    demands = np.array([info["demand"] for info in info_history])  # (T, N)
    inventories = np.array([info["inventory"] for info in info_history])  # (T, N)

    total_demand = demands.sum(axis=0)  # (N,)
    avg_inventory = inventories.mean(axis=0)  # (N,)

    # Clip to avoid division by near-zero inventory
    turnover = total_demand / np.maximum(avg_inventory, 1e-3)

    return turnover
