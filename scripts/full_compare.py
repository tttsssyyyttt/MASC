"""Full algorithm comparison: all RL + non-RL on all topologies.

Algorithms: BaseStock, (s,S), MILP, IQL, QMIX, MAPPO, SEA-IQL, SEA-QMIX, SEA-MAPPO
Topologies: 2layers, 3layers, 4layers
Config: hidden_dim=128, batch_size=64, GRU encoder, 200ep, 5 eval seeds

Usage (parallel per topology):
  python full_compare.py --topo 3layers
"""

import sys, os, time, json, argparse, warnings, csv, random

warnings.filterwarnings("ignore")
import sys
from pathlib import Path
# 将项目根目录加入 sys.path（假设 full_compare.py 在 meirp_project/scripts/ 下）
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
from tqdm import tqdm

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from core.config import from_yaml
from core.env import MEIRPEnv
from core.demand import DemandGenerator
from algorithms.registry import make_agent
from baselines.base_stock import BaseStockPolicy
from baselines.routed import RoutedBaseStockPolicy, RoutedSSPolicy, with_routing
from baselines.ss_policy import SSPolicy
from evaluation.metrics import compute_metrics
from collaboration.milp_advisor import MILPAdvisor
from collaboration.escalation import ThreeLevelEscalation
from solvers.ga_inventory_solver import GAInventorySolver
from solvers.alns_inventory_solver import ALNSInventorySolver
from solvers.alns_irp_solver import ALNSIRPSolver
from solvers.ortools_irp_solver import ORToolsIRPSolver


def set_global_seed(seed):
    """Reset global RNGs so one algorithm cannot affect the next."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "configs")
DEFAULT_RUN_CONFIG = os.path.join(CFG_DIR, "rl_pure.yaml")


def _metric_mean(values):
    first = values[0]
    if isinstance(first, dict):
        keys = sorted(set().union(*(v.keys() for v in values)))
        return {
            k: _metric_mean([v[k] for v in values if k in v])
            for k in keys
        }
    if isinstance(first, np.ndarray):
        return np.mean(np.stack(values, axis=0), axis=0)
    if isinstance(first, (int, float, np.integer, np.floating)):
        return float(np.mean(values))
    return values


def _metric_std(values):
    first = values[0]
    if isinstance(first, dict):
        keys = sorted(set().union(*(v.keys() for v in values)))
        return {
            k: _metric_std([v[k] for v in values if k in v])
            for k in keys
        }
    if isinstance(first, np.ndarray):
        return np.std(np.stack(values, axis=0), axis=0)
    if isinstance(first, (int, float, np.integer, np.floating)):
        return float(np.std(values))
    return None


def aggregate_metrics(metrics_list):
    """Aggregate all metric fields without dropping per-node/per-layer data."""
    if not metrics_list:
        return {}

    result = {}
    keys = sorted(set().union(*(m.keys() for m in metrics_list)))
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if not values:
            continue
        result[key] = _metric_mean(values)
        std = _metric_std(values)
        if std is not None:
            result[f"{key}_std"] = std

    result["n_eval_seeds"] = len(metrics_list)
    return result


def to_jsonable(obj):
    """Convert numpy-heavy metrics to JSON-native structures."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def eval_policy(
        agent,
        cfg,
        n_seeds=5,
        is_base=False,
        escalation=None,
        oracle_demand=False,
        return_raw=False,
        return_action_log=False,
        action_label=None,
        train_episode=None,
):
    ms = []
    action_rows = []
    for s in range(n_seeds):
        env = MEIRPEnv(cfg)
        demand_seq = None

        if oracle_demand:
            # Pre-generate TRUE terminal demands for the entire episode
            rng = np.random.default_rng(s)
            dgen = DemandGenerator(cfg, rng)
            T = cfg.episode_len
            demand_seq = np.array([dgen.generate(len(env.terminals)) for _ in range(T)], dtype=np.float64)
            # Pad with demand_mean beyond episode end for MILP oracle horizon
            pad_len = getattr(agent, 'planning_horizon', 15)
            pad = np.full((pad_len, len(env.terminals)), cfg.demand_mean, dtype=np.float64)
            demand_seq = np.concatenate([demand_seq, pad], axis=0)
            obs, _ = env.reset(seed=s, options={"demand_sequence": demand_seq[:T]})
        else:
            obs, _ = env.reset(seed=s)

        ih = []
        action_series = []
        done = False
        step = 0
        while not done:
            if is_base == "milp":
                if oracle_demand and demand_seq is not None:
                    horizon = getattr(agent, 'planning_horizon', 15)
                    future = demand_seq[step:step + horizon]
                else:
                    future = None
                a = agent.get_action(env.inventory, env.backlog, env.pipeline, demand_forecast=future)
            elif is_base:
                a = agent.get_actions(env)
            else:
                a = agent.get_actions(obs, deterministic=True)
            action_series.append(a)
            if escalation and not is_base:
                if oracle_demand and demand_seq is not None:
                    future = demand_seq[step:step + 15]
                    a, _, _ = escalation.step_eval(obs, a, env, demand_forecast=future)
                else:
                    a, _, _ = escalation.step(obs, a, env)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            ih.append(info)
            step += 1
        ms.append(compute_metrics(ih, cfg))
        if return_action_log:
            row = {
                "algorithm": action_label or "",
                "train_episode": train_episode if train_episode is not None else "",
                "eval_seed": s,
            }
            raw_summary = _summarize_actions(action_series, cfg)
            env_summary = _summarize_order_qty(ih, cfg)
            for key, value in raw_summary.items():
                row[f"eval_raw_{key}"] = value
            for key, value in env_summary.items():
                row[f"eval_env_{key}"] = value
            action_rows.append(row)

    aggregated = aggregate_metrics(ms)
    if return_raw and return_action_log:
        return aggregated, ms, action_rows
    if return_raw:
        return aggregated, ms
    if return_action_log:
        return aggregated, action_rows
    return aggregated


def _safe_float(value, default=""):
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return value
    return default


def _join_numbers(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return ";".join(f"{x:.6g}" for x in arr)


def _action_branch_values(action):
    if isinstance(action, dict):
        vals = [int(action.get("q", 0))]
        vals.extend(int(x) for x in action.get("alpha", []))
        return vals
    if isinstance(action, (list, tuple, np.ndarray)):
        return [int(x) for x in action]
    return [int(action)]


def _action_node_order(action):
    if isinstance(action, dict):
        return int(action.get("q", 0))
    return sum(_action_branch_values(action))


def _summarize_actions(action_series, cfg):
    if not action_series:
        return {}

    node_orders = []
    branch_values = []
    invalid_count = 0
    for actions in action_series:
        row = []
        for action in actions:
            vals = _action_branch_values(action)
            branch_values.extend(vals)
            invalid_count += sum(1 for x in vals if x < 0 or x > cfg.max_order)
            row.append(_action_node_order(action))
        node_orders.append(row)

    node_orders = np.asarray(node_orders, dtype=np.float64)
    branch_values = np.asarray(branch_values, dtype=np.float64)
    layers = np.asarray(cfg.layers)
    per_node_mean = node_orders.mean(axis=0)
    per_node_std = node_orders.std(axis=0)
    per_layer_mean = []
    for layer in sorted(set(cfg.layers)):
        per_layer_mean.append(float(per_node_mean[layers == layer].mean()))

    return {
        "mean_action": float(node_orders.mean()),
        "std_action": float(node_orders.std()),
        "zero_action_rate": float(np.mean(branch_values == 0)) if branch_values.size else 0.0,
        "max_action_rate": float(np.mean(branch_values >= cfg.max_order)) if branch_values.size else 0.0,
        "invalid_action_count": int(invalid_count),
        "per_node_mean_action": _join_numbers(per_node_mean),
        "per_node_std_action": _join_numbers(per_node_std),
        "per_layer_mean_action": _join_numbers(per_layer_mean),
    }


def _summarize_order_qty(info_history, cfg):
    if not info_history:
        return {}
    orders = np.asarray([info["order_qty"] for info in info_history], dtype=np.float64)
    layers = np.asarray(cfg.layers)
    per_node_mean = orders.mean(axis=0)
    per_node_std = orders.std(axis=0)
    per_layer_mean = []
    for layer in sorted(set(cfg.layers)):
        per_layer_mean.append(float(per_node_mean[layers == layer].mean()))

    return {
        "mean_order_qty": float(orders.mean()),
        "std_order_qty": float(orders.std()),
        "zero_order_rate": float(np.mean(orders <= 1e-9)),
        "max_order_rate": float(np.mean(orders >= cfg.max_order - 1e-9)),
        "per_node_mean_order_qty": _join_numbers(per_node_mean),
        "per_node_std_order_qty": _join_numbers(per_node_std),
        "per_layer_mean_order_qty": _join_numbers(per_layer_mean),
    }


def _mean_update_metrics(update_metrics):
    keys = sorted(set().union(*(m.keys() for m in update_metrics))) if update_metrics else []
    result = {}
    for key in keys:
        vals = [_safe_float(m.get(key), None) for m in update_metrics]
        vals = [v for v in vals if v is not None]
        if vals:
            result[key] = float(np.mean(vals))
    return result


def _attention_summary(agent):
    encoder = getattr(agent, "encoder", None)
    gat = getattr(encoder, "gat", None)
    if gat is None:
        return None
    weights = gat.get_attention_weights()
    if weights is None:
        return None
    arr = weights.detach().cpu().numpy().astype(np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    mass = np.abs(arr)
    denom = mass.sum()
    if denom > 1e-12:
        p = mass / denom
        entropy = float(-(p * np.log(p + 1e-12)).sum())
    else:
        entropy = 0.0
    return {
        "attention_mean": float(arr.mean()),
        "attention_std": float(arr.std()),
        "attention_min": float(arr.min()),
        "attention_max": float(arr.max()),
        "attention_entropy": entropy,
        "attention_edges": int(arr.size),
    }


def _copy_scalar_metrics(row, metrics, prefix=""):
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            row[f"{prefix}{key}"] = _safe_float(value)


def _annotate_cost_metrics(metrics, cfg, name):
    routing_enabled = bool(getattr(cfg, "routing_enabled", False))
    if name == "MILP-InventoryOnly":
        scope = "inventory_only_oracle"
    elif routing_enabled:
        scope = "inventory_plus_routing"
    else:
        scope = "inventory_only"

    metrics["cost_scope"] = scope
    metrics["routing_enabled"] = routing_enabled
    metrics["objective_formula"] = "weighted_total_cost = total_inventory_cost + transport_weight * route_distance"
    metrics["transport_weight"] = float(getattr(cfg, "transport_cost", 0.0))
    n_steps = float(metrics.get("n_steps", 0) or 0)
    if n_steps > 0:
        if "total_inventory_cost" in metrics:
            metrics["avg_inventory_cost_per_step"] = float(metrics["total_inventory_cost"]) / n_steps
        if "total_transport" in metrics:
            metrics["routing_cost"] = float(metrics["total_transport"])
            metrics["avg_routing_cost_per_step"] = float(metrics["total_transport"]) / n_steps
        if "weighted_total_cost" not in metrics:
            inv = float(metrics.get("total_inventory_cost", 0.0))
            route = float(metrics.get("total_transport", 0.0))
            metrics["weighted_total_cost"] = inv + route
            metrics["avg_weighted_total_cost_per_step"] = metrics["weighted_total_cost"] / n_steps


def train_rl(
        agent,
        cfg,
        escalation,
        n_ep,
        warmup,
        label,
        eval_interval=0,
        eval_seeds=1,
        eval_mode="pure",
):
    env = MEIRPEnv(cfg)
    R = []
    steps = 0
    train_rows = []
    eval_rows = []
    action_rows = []
    eval_action_rows = []
    attention_rows = []
    for ep in tqdm(range(n_ep)):
        obs, _ = env.reset()
        ep_r = 0.0
        done = False
        episode_infos = []
        rl_actions = []
        executed_actions = []
        update_metrics = []
        pipeline_levels = []
        while not done:
            rl_act = agent.get_actions(obs)
            if escalation:
                a, src, info = escalation.step(obs, rl_act, env)
            else:
                a = rl_act
            nobs, r, term, trunc, info = env.step(a)
            done = term or trunc
            rl_actions.append(rl_act)
            executed_actions.append(a)
            episode_infos.append(info)
            pipeline_levels.append(float(env.pipeline.mean()))
            # Always store original RL action (expert only helps execute better trajectories)
            agent.store_transition(obs, rl_act, r, nobs, np.full(cfg.num_nodes, float(term)))
            steps += 1
            # Off-policy (IQL, QMIX): step-based update when buffer has enough samples
            if hasattr(agent, 'batch_size') and len(agent.buffer) >= agent.batch_size and steps >= warmup:
                update_metrics.append(agent.update())
            ep_r += float(r.sum())
            obs = nobs
        # On-policy (MAPPO): episode-based update
        if not hasattr(agent, 'batch_size') and steps >= warmup:
            update_metrics.append(agent.update())
        R.append(ep_r)

        ep_metrics = compute_metrics(episode_infos, cfg) if episode_infos else {}
        _annotate_cost_metrics(ep_metrics, cfg, label)
        mean_updates = _mean_update_metrics(update_metrics)
        train_row = {
            "algorithm": label,
            "episode": ep + 1,
            "global_step": steps,
            "episode_reward_sum": float(ep_r),
            "episode_reward_mean_per_step": float(ep_r / max(len(episode_infos), 1)),
            "episode_length": len(episode_infos),
            "update_count": len(update_metrics),
            "epsilon": _safe_float(agent._get_epsilon()) if hasattr(agent, "_get_epsilon") else "",
            "buffer_size": len(agent.buffer) if hasattr(agent, "buffer") else "",
            "avg_pipeline": float(np.mean(pipeline_levels)) if pipeline_levels else 0.0,
        }
        for key in ("loss", "policy_loss", "value_loss", "entropy"):
            train_row[key] = _safe_float(mean_updates.get(key, ""))
        for key in (
                "avg_cost_per_step",
                "weighted_total_cost",
                "avg_weighted_total_cost_per_step",
                "total_inventory_cost",
                "avg_inventory_cost_per_step",
                "total_transport",
                "routing_cost",
                "avg_routing_cost_per_step",
                "transport_weight",
                "avg_transport_per_step",
                "avg_route_distance_per_step",
                "routing_feasible_rate",
                "avg_fill_rate",
                "avg_stockout_rate",
                "avg_bullwhip_cv",
                "total_holding",
                "total_backlog",
        ):
            train_row[key] = _safe_float(ep_metrics.get(key, ""))
        train_rows.append(train_row)

        rl_summary = _summarize_actions(rl_actions, cfg)
        exec_summary = _summarize_actions(executed_actions, cfg)
        env_order_summary = _summarize_order_qty(episode_infos, cfg)
        action_row = {"algorithm": label, "episode": ep + 1}
        for key, value in rl_summary.items():
            action_row[f"rl_{key}"] = value
        for key, value in exec_summary.items():
            action_row[f"exec_{key}"] = value
        for key, value in env_order_summary.items():
            action_row[f"env_{key}"] = value
        action_rows.append(action_row)

        attn = _attention_summary(agent)
        if attn is not None:
            attn["algorithm"] = label
            attn["episode"] = ep + 1
            attention_rows.append(attn)

        if eval_interval > 0 and (ep + 1) % eval_interval == 0:
            eval_kwargs = {}
            if eval_mode == "sea":
                eval_kwargs = {
                    "escalation": ThreeLevelEscalation(cfg, k_spc=3.0),
                    "oracle_demand": True,
                }
            eval_metrics, raw_eval, interval_eval_actions = eval_policy(
                agent,
                cfg,
                n_seeds=eval_seeds,
                return_raw=True,
                return_action_log=True,
                action_label=label,
                train_episode=ep + 1,
                **eval_kwargs,
            )
            _annotate_cost_metrics(eval_metrics, cfg, label)
            eval_row = {
                "algorithm": label,
                "train_episode": ep + 1,
                "eval_seeds": eval_seeds,
            }
            _copy_scalar_metrics(eval_row, eval_metrics)
            eval_rows.append(eval_row)
            eval_action_rows.extend(interval_eval_actions)

        if (ep + 1) % 100 == 0:
            print(f"    [{label}] Ep{ep + 1}/{n_ep} R={np.mean(R[-50:]):.1f}", flush=True)
    return R, train_rows, eval_rows, action_rows, eval_action_rows, attention_rows


def _summary_value(metrics, key):
    value = metrics.get(key, "")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return value


def write_summary_csv(path, order, results, runtimes):
    fields = [
        "algorithm",
        "cost_scope",
        "transport_weight",
        "avg_inventory_cost_per_step",
        "avg_routing_cost_per_step",
        "avg_weighted_total_cost_per_step",
        "avg_cost_per_step",
        "avg_fill_rate",
        "avg_stockout_rate",
        "avg_bullwhip_cv",
        "avg_requested_bullwhip_cv",
        "avg_shipped_bullwhip_cv",
        "max_bullwhip_cv",
        "max_requested_bullwhip_cv",
        "avg_order_mean",
        "avg_order_std",
        "avg_order_var",
        "avg_demand_mean",
        "avg_demand_std",
        "avg_demand_var",
        "avg_requested_demand_mean",
        "avg_requested_demand_std",
        "avg_requested_demand_var",
        "avg_shipped_demand_mean",
        "avg_shipped_demand_std",
        "avg_shipped_demand_var",
        "avg_fulfilled_demand_mean",
        "avg_fulfilled_demand_std",
        "avg_fulfilled_demand_var",
        "min_demand_var",
        "min_requested_demand_var",
        "max_order_var",
        "avg_transport_per_step",
        "avg_route_distance_per_step",
        "routing_feasible_rate",
        "runtime_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name in order:
            metrics = results[name]
            row = {"algorithm": name, "runtime_sec": runtimes.get(name, 0.0)}
            for key in fields[1:-1]:
                row[key] = _summary_value(metrics, key)
            writer.writerow(row)


def write_rows_csv(path, rows, preferred_fields=None):
    if not rows:
        return False
    preferred_fields = preferred_fields or []
    keys = sorted(set().union(*(row.keys() for row in rows)))
    fields = [k for k in preferred_fields if k in keys]
    fields.extend(k for k in keys if k not in fields)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return True


def write_curves_json(path, curves):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(curves), f, indent=2)


def plot_convergence_curves(curves, path):
    if not curves:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, rewards in curves.items():
        if not rewards:
            continue
        rewards_arr = np.array(rewards, dtype=np.float64)
        window = min(20, max(1, len(rewards_arr) // 5))
        if window > 1:
            y = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, window + len(y)), y, label=name)
        else:
            ax.plot(np.arange(1, len(rewards_arr) + 1), rewards_arr, label=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("RL Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--topo", required=True, choices=["2layers", "3layers", "4layers", "custom"])
    p.add_argument("--ep", type=int, default=300)
    p.add_argument("--hard", action="store_true")
    p.add_argument("--config", type=str, default=None, help="Direct path to config YAML")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--transport-cost", type=float, default=1.0, help="Cost per routing distance unit for routed baselines")
    p.add_argument("--transport-weight", type=float, default=None, help="Alias for --transport-cost; records the route-distance weight explicitly")
    p.add_argument("--vehicle-capacity", type=float, default=None, help="Vehicle capacity for routed baselines")
    p.add_argument("--vehicle-count", type=int, default=None, help="Vehicles per depot for routed baselines")
    p.add_argument("--ga-pop", type=int, default=16, help="GA inventory population size")
    p.add_argument("--ga-gen", type=int, default=10, help="GA inventory generations")
    p.add_argument("--alns-iter", type=int, default=30, help="ALNS inventory iterations")
    p.add_argument("--ortools-horizon", type=int, default=5, help="OR-Tools rolling horizon")
    p.add_argument("--include-ortools", action="store_true", help="Include OR-Tools baseline (can be slow or unstable on some OR-Tools builds)")
    p.add_argument("--eval-seeds", type=int, default=5, help="Number of evaluation seeds")
    p.add_argument("--skip-rl", action="store_true", help="Run only non-RL baselines")
    p.add_argument("--skip-milp", action="store_true", help="Skip MILP advisor baseline")
    p.add_argument("--skip-sea", action="store_true", help="Skip SEA-assisted RL variants")
    p.add_argument("--include-gnn-rl", action="store_true", help="Include IQL/QMIX/MAPPO with GNN/GAT encoders")
    p.add_argument("--rl-eval-interval", type=int, default=50, help="Evaluate RL agents every N training episodes; 0 disables interval eval")
    p.add_argument("--rl-eval-seeds", type=int, default=1, help="Evaluation seeds used for interval RL diagnostics")
    p.add_argument("--output-prefix", type=str, default=None, help="Output filename prefix under results/")
    return p


def run_compare(args):
    if args.transport_weight is not None:
        args.transport_cost = args.transport_weight

    if args.config:
        cfg = from_yaml(args.config)
    else:
        suffix = "_hard" if args.hard else ""
        cfg = from_yaml(os.path.join(CFG_DIR, f"env_{args.topo}{suffix}.yaml"))
    set_global_seed(cfg.seed)
    obs_dim = 6 + 3 * cfg.lead_time
    N = cfg.num_nodes
    print(f"\n{'=' * 60}\n  {args.topo}: {N} nodes | max_order={cfg.max_order} | demand={cfg.demand_mean}\n{'=' * 60}")

    results = {}
    raw_results = {}
    runtimes = {}
    order = []
    training_curves = {}
    rl_train_logs = []
    rl_eval_logs = []
    rl_action_logs = []
    rl_eval_action_logs = []
    rl_attention_logs = []

    # ===== Non-RL baselines =====
    baseline_specs = [
        ("BaseStock", cfg, BaseStockPolicy(cfg), True, False),
        ("(s,S)", cfg, SSPolicy(cfg), True, False),
    ]

    routed_specs = [
        ("BaseStock+Savings", "savings", False, RoutedBaseStockPolicy),
        ("(s,S)+Savings", "savings", False, RoutedSSPolicy),
        ("BaseStock+Savings2Opt", "savings", True, RoutedBaseStockPolicy),
        ("(s,S)+Savings2Opt", "savings", True, RoutedSSPolicy),
        ("BaseStock+NearestInsertion", "nearest_insertion", False, RoutedBaseStockPolicy),
        ("(s,S)+NearestInsertion", "nearest_insertion", False, RoutedSSPolicy),
        ("BaseStock+GARouting", "ga", True, RoutedBaseStockPolicy),
        ("(s,S)+GARouting", "ga", True, RoutedSSPolicy),
    ]
    for name, method, use_2opt, policy_cls in routed_specs:
        rcfg = with_routing(
            cfg,
            method=method,
            use_2opt=use_2opt,
            transport_cost=args.transport_cost,
            vehicle_capacity=args.vehicle_capacity,
            vehicle_count=args.vehicle_count,
        )
        baseline_specs.append((name, rcfg, policy_cls(rcfg, method), True, False))

    metaheuristic_specs = [
        ("GAInventory+Savings", "savings", False, "ga"),
        ("GAInventory+Savings2Opt", "savings", True, "ga"),
        ("ALNSInventory+Savings", "savings", False, "alns"),
        ("ALNSInventory+Savings2Opt", "savings", True, "alns"),
    ]
    for name, method, use_2opt, solver_name in metaheuristic_specs:
        rcfg = with_routing(
            cfg,
            method=method,
            use_2opt=use_2opt,
            transport_cost=args.transport_cost,
            vehicle_capacity=args.vehicle_capacity,
            vehicle_count=args.vehicle_count,
        )
        if solver_name == "ga":
            policy = GAInventorySolver(
                rcfg,
                population_size=args.ga_pop,
                n_generations=args.ga_gen,
                planning_horizon=5,
            )
        else:
            policy = ALNSInventorySolver(
                rcfg,
                max_iterations=args.alns_iter,
                planning_horizon=5,
            )
        baseline_specs.append((name, rcfg, policy, True, False))

    irp_cfg = with_routing(
        cfg,
        method="savings",
        use_2opt=True,
        transport_cost=args.transport_cost,
        vehicle_capacity=args.vehicle_capacity,
        vehicle_count=args.vehicle_count,
    )
    baseline_specs.append((
        "ALNS-IRP",
        irp_cfg,
        ALNSIRPSolver(irp_cfg, max_iterations=args.alns_iter, planning_horizon=5),
        True,
        False,
    ))

    if args.include_ortools:
        baseline_specs.append((
            "ORTools-IRP",
            irp_cfg,
            ORToolsIRPSolver(irp_cfg, planning_horizon=args.ortools_horizon, time_limit=5.0),
            True,
            False,
        ))

    if not args.skip_milp:
        baseline_specs.append(("MILP-InventoryOnly", cfg, MILPAdvisor(cfg, planning_horizon=15), False, True))

    for name, eval_cfg, policy, base, oracle in baseline_specs:
        t0 = time.time()
        is_base_flag = "milp" if oracle else base
        r, raw = eval_policy(
            policy,
            eval_cfg,
            n_seeds=args.eval_seeds,
            is_base=is_base_flag,
            oracle_demand=oracle,
            return_raw=True,
        )
        _annotate_cost_metrics(r, eval_cfg, name)
        for raw_metric in raw:
            _annotate_cost_metrics(raw_metric, eval_cfg, name)
        runtimes[name] = time.time() - t0
        results[name] = r
        raw_results[name] = raw
        order.append(name)
        route = r.get("avg_route_distance_per_step", 0.0)
        print(f"  {name:<28}: cost={r['avg_cost_per_step']:.1f} fill={r['avg_fill_rate']:.4f} bw={r['avg_bullwhip_cv']:.4f} route={route:.2f}")

    # ===== RL algorithms =====
    rl_configs = [
        ("QMIX", "qmix", False, dict(hidden_dim=128, lr=5e-5, gamma=0.95, buffer_capacity=20000, batch_size=64, target_update_freq=500, mixer_hidden=32)),
        ("IQL", "iql", False, dict(hidden_dim=128, lr=5e-4, gamma=0.99, buffer_capacity=20000, batch_size=64, epsilon_decay=30000, target_update_freq=200, encoder_update_freq=8)),
        # ("MAPPO", "mappo", False, dict(hidden_dim=128, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, entropy_coef=0.01)),
    ]
    if args.include_gnn_rl:
        rl_configs.extend([
            ("IQL-GNN", "iql", True, dict(hidden_dim=128, lr=3e-4, gamma=0.99, buffer_capacity=20000, batch_size=64, epsilon_decay=12000, target_update_freq=200, encoder_update_freq=8)),
            ("QMIX-GNN", "qmix", True, dict(hidden_dim=128, lr=3e-4, gamma=0.99, buffer_capacity=20000, batch_size=64, target_update_freq=100, mixer_hidden=32, encoder_update_freq=4)),
            ("MAPPO-GNN", "mappo", True, dict(hidden_dim=128, lr_actor=2e-4, lr_critic=8e-4, gamma=0.99, entropy_coef=0.01)),
        ])

    if args.skip_rl:
        rl_configs = []

    for name, algo, use_gnn, kw in rl_configs:
        # Pure RL
        print(f"\n  [{name}] training {args.ep}ep...")
        set_global_seed(cfg.seed)
        agent = make_agent(algo, cfg, obs_dim=obs_dim, device=args.device, use_gnn=use_gnn, **kw)
        t0 = time.time()

        warmup_steps = 0 if algo == "mappo" else 500

        rewards, train_log, eval_log, action_log, eval_action_log, attention_log = train_rl(
            agent,
            cfg,
            None,
            args.ep,
            warmup_steps,
            name,
            eval_interval=args.rl_eval_interval,
            eval_seeds=args.rl_eval_seeds,
            eval_mode="pure",
        )
        t_pure = time.time() - t0
        training_curves[name] = [float(x) for x in rewards]
        rl_train_logs.extend(train_log)
        rl_eval_logs.extend(eval_log)
        rl_action_logs.extend(action_log)
        rl_eval_action_logs.extend(eval_action_log)
        rl_attention_logs.extend(attention_log)
        r, raw, final_eval_actions = eval_policy(
            agent,
            cfg,
            n_seeds=args.eval_seeds,
            return_raw=True,
            return_action_log=True,
            action_label=name,
            train_episode=args.ep,
        )
        rl_eval_action_logs.extend(final_eval_actions)
        _annotate_cost_metrics(r, cfg, name)
        for raw_metric in raw:
            _annotate_cost_metrics(raw_metric, cfg, name)
        runtimes[name] = t_pure
        results[name] = r
        raw_results[name] = raw
        order.append(name)
        print(f"    {name:<12}: cost={r['avg_cost_per_step']:.1f} fill={r['avg_fill_rate']:.4f} bw={r['avg_bullwhip_cv']:.4f} time={t_pure:.0f}s")

        if args.skip_sea:
            continue

        # SEA: k=3.0 for both training and eval
        set_global_seed(cfg.seed)
        sea_agent = make_agent(algo, cfg, obs_dim=obs_dim, device=args.device, use_gnn=use_gnn, **kw)
        esc_train = ThreeLevelEscalation(cfg, k_spc=3.0)
        t0 = time.time()

        warmup_steps = 0 if algo == "mappo" else 500

        sea_rewards, train_log, eval_log, action_log, eval_action_log, attention_log = train_rl(
            sea_agent,
            cfg,
            esc_train,
            args.ep,
            warmup_steps,
            f"SEA-{name}",
            eval_interval=args.rl_eval_interval,
            eval_seeds=args.rl_eval_seeds,
            eval_mode="sea",
        )
        t_sea = time.time() - t0
        training_curves[f"SEA-{name}"] = [float(x) for x in sea_rewards]
        rl_train_logs.extend(train_log)
        rl_eval_logs.extend(eval_log)
        rl_action_logs.extend(action_log)
        rl_eval_action_logs.extend(eval_action_log)
        rl_attention_logs.extend(attention_log)
        esc_eval = ThreeLevelEscalation(cfg, k_spc=3.0)
        r_sea, raw_sea, final_eval_actions = eval_policy(
            sea_agent,
            cfg,
            n_seeds=args.eval_seeds,
            escalation=esc_eval,
            oracle_demand=True,
            return_raw=True,
            return_action_log=True,
            action_label=f"SEA-{name}",
            train_episode=args.ep,
        )
        sname = f"SEA-{name}"
        rl_eval_action_logs.extend(final_eval_actions)
        _annotate_cost_metrics(r_sea, cfg, sname)
        for raw_metric in raw_sea:
            _annotate_cost_metrics(raw_metric, cfg, sname)
        results[sname] = r_sea
        raw_results[sname] = raw_sea
        runtimes[sname] = t_sea
        order.append(sname)
        s = esc_train.stats()
        print(f"    {sname:<12}: cost={r_sea['avg_cost_per_step']:.1f} fill={r_sea['avg_fill_rate']:.4f} bw={r_sea['avg_bullwhip_cv']:.4f} time={t_sea:.0f}s SPC={s['spc']:.2%} Exp={s['expert']:.2%}")

    # ===== Summary table =====
    print(f"\n  {'=' * 70}")
    print(f"  Results Summary: {args.topo}")
    print(f"  {'Algorithm':<16} {'Cost/Step':>10} {'Fill Rate':>10} {'Bullwhip':>10} {'Stockout':>10}")
    print(f"  {'-' * 56}")
    best_fill = max(results[n]["avg_fill_rate"] for n in order)
    best_cost = min(results[n]["avg_cost_per_step"] for n in order)
    for name in order:
        r = results[name]
        fm = " *" if abs(r["avg_fill_rate"] - best_fill) < 1e-6 else ""
        cm = " *" if abs(r["avg_cost_per_step"] - best_cost) < 1e-6 else ""
        print(f"  {name:<16} {r['avg_cost_per_step']:>10.1f}{cm} {r['avg_fill_rate']:>10.4f}{fm} {r['avg_bullwhip_cv']:>10.4f} {r['avg_stockout_rate']:>10.4f}")

    # ===== Save =====
    prefix = args.output_prefix or f"full_compare_{args.topo}"
    out_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
    out = os.path.join(out_dir, f"{prefix}.json")
    csv_out = os.path.join(out_dir, f"{prefix}_summary.csv")
    curves_out = os.path.join(out_dir, f"{prefix}_curves.json")
    curves_png = os.path.join(out_dir, f"{prefix}_convergence.png")
    rl_train_out = os.path.join(out_dir, f"{prefix}_rl_train_log.csv")
    rl_eval_out = os.path.join(out_dir, f"{prefix}_rl_eval_log.csv")
    rl_action_out = os.path.join(out_dir, f"{prefix}_rl_action_log.csv")
    rl_eval_action_out = os.path.join(out_dir, f"{prefix}_rl_eval_action_log.csv")
    rl_attention_out = os.path.join(out_dir, f"{prefix}_gnn_attention_log.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(to_jsonable({
            "topo": args.topo,
            "config": args.config,
            "eval_seeds": args.eval_seeds,
            "skip_rl": args.skip_rl,
            "transport_weight": args.transport_cost,
            "objective_formula": "weighted_total_cost = total_inventory_cost + transport_weight * route_distance",
            "milp_note": "MILP-InventoryOnly is an inventory-only oracle/advisor baseline and does not include routing cost.",
            "results": results,
            "raw_results": raw_results,
            "runtimes": runtimes,
            "order": order,
            "training_curves": training_curves,
            "diagnostic_files": {
                "rl_train_log": rl_train_out if rl_train_logs else None,
                "rl_eval_log": rl_eval_out if rl_eval_logs else None,
                "rl_action_log": rl_action_out if rl_action_logs else None,
                "rl_eval_action_log": rl_eval_action_out if rl_eval_action_logs else None,
                "gnn_attention_log": rl_attention_out if rl_attention_logs else None,
            },
        }), f, indent=2)
    write_summary_csv(csv_out, order, results, runtimes)
    if write_rows_csv(
            rl_train_out,
            rl_train_logs,
            preferred_fields=[
                "algorithm", "episode", "global_step", "episode_reward_sum",
                "episode_reward_mean_per_step", "episode_length", "update_count",
                "epsilon", "buffer_size", "loss", "policy_loss", "value_loss",
                "entropy", "avg_cost_per_step", "avg_weighted_total_cost_per_step",
                "avg_inventory_cost_per_step", "avg_routing_cost_per_step",
                "transport_weight", "total_inventory_cost",
                "total_transport", "routing_cost", "avg_transport_per_step",
                "avg_route_distance_per_step", "routing_feasible_rate",
                "avg_fill_rate", "avg_stockout_rate", "avg_bullwhip_cv",
                "avg_requested_bullwhip_cv", "avg_shipped_bullwhip_cv",
                "max_bullwhip_cv", "avg_order_mean", "avg_order_std",
                "avg_order_var", "avg_demand_mean", "avg_demand_std",
                "avg_demand_var", "avg_requested_demand_mean",
                "avg_requested_demand_std", "avg_requested_demand_var",
                "avg_shipped_demand_mean", "avg_shipped_demand_std",
                "avg_shipped_demand_var", "avg_fulfilled_demand_mean",
                "avg_fulfilled_demand_std", "avg_fulfilled_demand_var",
                "min_demand_var",
                "min_requested_demand_var", "max_order_var",
                "total_holding", "total_backlog", "avg_pipeline",
            ],
    ):
        print(f"  Saved: {rl_train_out}")
    if write_rows_csv(
            rl_eval_out,
            rl_eval_logs,
            preferred_fields=[
                "algorithm", "train_episode", "eval_seeds", "avg_cost_per_step",
                "avg_weighted_total_cost_per_step", "avg_inventory_cost_per_step",
                "avg_routing_cost_per_step", "transport_weight",
                "total_inventory_cost", "total_transport", "routing_cost",
                "avg_transport_per_step",
                "avg_route_distance_per_step", "routing_feasible_rate",
                "avg_fill_rate", "avg_stockout_rate", "avg_bullwhip_cv",
                "avg_requested_bullwhip_cv", "avg_shipped_bullwhip_cv",
                "max_bullwhip_cv", "avg_order_mean", "avg_order_std",
                "avg_order_var", "avg_demand_mean", "avg_demand_std",
                "avg_demand_var", "avg_requested_demand_mean",
                "avg_requested_demand_std", "avg_requested_demand_var",
                "avg_shipped_demand_mean", "avg_shipped_demand_std",
                "avg_shipped_demand_var", "avg_fulfilled_demand_mean",
                "avg_fulfilled_demand_std", "avg_fulfilled_demand_var",
                "min_demand_var",
                "min_requested_demand_var", "max_order_var",
                "total_holding", "total_backlog",
            ],
    ):
        print(f"  Saved: {rl_eval_out}")
    if write_rows_csv(
            rl_action_out,
            rl_action_logs,
            preferred_fields=[
                "algorithm", "episode", "rl_mean_action", "rl_std_action",
                "rl_zero_action_rate", "rl_max_action_rate",
                "rl_invalid_action_count", "exec_mean_action", "exec_std_action",
                "exec_zero_action_rate", "exec_max_action_rate",
                "exec_invalid_action_count", "env_mean_order_qty",
                "env_std_order_qty", "env_zero_order_rate", "env_max_order_rate",
                "rl_per_node_mean_action",
                "rl_per_node_std_action", "rl_per_layer_mean_action",
                "exec_per_node_mean_action", "exec_per_node_std_action",
                "exec_per_layer_mean_action", "env_per_node_mean_order_qty",
                "env_per_node_std_order_qty", "env_per_layer_mean_order_qty",
            ],
    ):
        print(f"  Saved: {rl_action_out}")
    if write_rows_csv(
            rl_eval_action_out,
            rl_eval_action_logs,
            preferred_fields=[
                "algorithm", "train_episode", "eval_seed",
                "eval_raw_mean_action", "eval_raw_std_action",
                "eval_raw_zero_action_rate", "eval_raw_max_action_rate",
                "eval_env_mean_order_qty", "eval_env_std_order_qty",
                "eval_env_zero_order_rate", "eval_env_max_order_rate",
                "eval_raw_per_node_mean_action", "eval_raw_per_node_std_action",
                "eval_raw_per_layer_mean_action",
                "eval_env_per_node_mean_order_qty",
                "eval_env_per_node_std_order_qty",
                "eval_env_per_layer_mean_order_qty",
            ],
    ):
        print(f"  Saved: {rl_eval_action_out}")
    if write_rows_csv(
            rl_attention_out,
            rl_attention_logs,
            preferred_fields=[
                "algorithm", "episode", "attention_mean", "attention_std",
                "attention_min", "attention_max", "attention_entropy",
                "attention_edges",
            ],
    ):
        print(f"  Saved: {rl_attention_out}")
    if training_curves:
        write_curves_json(curves_out, training_curves)
        if plot_convergence_curves(training_curves, curves_png):
            print(f"  Saved: {curves_png}")
    print(f"\n  Saved: {out}")
    print(f"  Saved: {csv_out}")
    if training_curves:
        print(f"  Saved: {curves_out}")


def args_from_run_config(path):
    """Load experiment run options from YAML."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "env_config" in raw and "config" not in raw:
        raw["config"] = raw["env_config"]

    defaults = {
        "topo": "3layers",
        "ep": 300,
        "hard": False,
        "config": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "transport_cost": 1.0,
        "transport_weight": None,
        "vehicle_capacity": None,
        "vehicle_count": None,
        "ga_pop": 16,
        "ga_gen": 10,
        "alns_iter": 30,
        "ortools_horizon": 5,
        "include_ortools": False,
        "eval_seeds": 5,
        "skip_rl": False,
        "skip_milp": False,
        "skip_sea": False,
        "include_gnn_rl": False,
        "rl_eval_interval": 50,
        "rl_eval_seeds": 1,
        "output_prefix": None,
    }
    values = {}
    for key, default in defaults.items():
        values[key] = raw.get(key, default)

    if values["device"] == "auto":
        values["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ep: {values['ep']}")
    return argparse.Namespace(**values)


def run_from_config(path=DEFAULT_RUN_CONFIG):
    """Run full comparison from a project config file."""
    args = args_from_run_config(path)
    print(f"Using run config: {path}", flush=True)
    run_compare(args)


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_compare(args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        run_from_config()
