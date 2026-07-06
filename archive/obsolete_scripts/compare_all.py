"""Compare ALL algorithms across ALL metrics on ALL topologies.

Algorithms: BaseStock, (s,S), IQL, QMIX, MAPPO, IQL+GNN, QMIX+GNN, MAPPO+GNN
Metrics: total_cost, avg_cost_per_step, total_holding, total_backlog,
         avg_fill_rate, min_fill_rate, avg_bullwhip_cv, avg_inventory_turnover
Topologies: 2-layer, 3-layer, 4-layer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.registry import make_agent
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.baselines.ss_policy import SSPolicy
from meirp_project.evaluation.metrics import compute_metrics


def train_agent(agent, cfg, n_episodes=500, warmup_steps=1000, log_interval=50):
    """Train a MADRL agent on the given env config.

    Supports both off-policy (IQL, QMIX) and on-policy (MAPPO) training.
    """
    is_on_policy = hasattr(agent, 'buffer') and hasattr(agent.buffer, 'compute_returns')

    for ep in range(n_episodes):
        env = MEIRPEnv(cfg)
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        done = False

        while not done:
            actions = agent.get_actions(obs)

            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            if hasattr(agent, 'store_transition'):
                dones = np.full(cfg.num_nodes, float(terminated))
                agent.store_transition(obs, actions, rewards, next_obs, dones)

            ep_reward += rewards.sum()
            obs = next_obs

            # Off-policy: update every step after warmup
            if not is_on_policy and hasattr(agent, 'update'):
                if getattr(agent, 'total_steps', 0) >= warmup_steps:
                    agent.update()

        # On-policy: update at end of episode
        if is_on_policy and hasattr(agent, 'update'):
            if hasattr(agent.buffer, 'advantages') and agent.buffer.advantages:
                agent.update()
            elif len(getattr(agent.buffer, 'rewards', [])) > 0:
                agent.update()

        if (ep + 1) % log_interval == 0:
            print(f"    Episode {ep+1}/{n_episodes} | Reward: {ep_reward:.1f}")


def run_episode_baseline(policy, cfg, seed=0):
    """Run one episode with a baseline policy, return metrics."""
    env = MEIRPEnv(cfg)
    obs, _ = env.reset(seed=seed)
    info_history = []
    done = False
    while not done:
        actions = policy.get_actions(env)
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        info_history.append(info)
    return compute_metrics(info_history, cfg)


def run_episode_agent(agent, cfg, seed=0, deterministic=True):
    """Run one episode with a MADRL agent, return metrics."""
    env = MEIRPEnv(cfg)
    obs, _ = env.reset(seed=seed)
    info_history = []
    done = False
    while not done:
        actions = agent.get_actions(obs, deterministic=deterministic)
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        info_history.append(info)
    return compute_metrics(info_history, cfg)


def multi_run(func, n_seeds=5):
    """Run func with multiple seeds, aggregate mean +/- std."""
    all_results = [func(s) for s in range(n_seeds)]
    keys = all_results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in all_results if k in r]
        if not vals:
            continue
        if isinstance(vals[0], (int, float)):
            agg[k] = f"{np.mean(vals):.2f} +/- {np.std(vals):.2f}"
            agg[f"{k}_mean"] = np.mean(vals)
            agg[f"{k}_std"] = np.std(vals)
        elif isinstance(vals[0], np.ndarray):
            agg[k] = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
            agg[f"{k}_mean"] = np.mean(vals)
            agg[f"{k}_std"] = np.std(vals)
    return agg


def main():
    configs = {
        "2-layer": "meirp_project/configs/env_2layers.yaml",
        "3-layer": "meirp_project/configs/env_3layers.yaml",
        "4-layer": "meirp_project/configs/env_4layers.yaml",
    }

    # Key metrics to display
    display_keys = [
        "total_cost", "total_holding", "total_backlog",
        "avg_fill_rate", "min_fill_rate",
        "avg_bullwhip_cv", "avg_inventory_turnover",
    ]

    n_seeds = 3  # number of evaluation seeds
    results_all = {}
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    for topo_name, cfg_path in configs.items():
        log(f"\n{'='*80}")
        log(f"  Topology: {topo_name}  ({cfg_path})")
        log(f"{'='*80}")
        cfg = from_yaml(cfg_path)
        obs_dim = 3 + 3 * cfg.lead_time
        results_all[topo_name] = {}

        # ---- Baselines ----
        baselines = {
            "BaseStock": lambda s: run_episode_baseline(BaseStockPolicy(cfg), cfg, seed=s),
            "(s,S)": lambda s: run_episode_baseline(SSPolicy(cfg), cfg, seed=s),
        }

        # ---- MADRL Algorithms ----
        algo_configs = [
            ("IQL", "iql", False),
            ("QMIX", "qmix", False),
            ("MAPPO", "mappo", False),
            ("IQL+GNN", "iql", True),
            ("QMIX+GNN", "qmix", True),
            ("MAPPO+GNN", "mappo", True),
        ]

        algo_runners = {}
        for display_name, algo_name, use_gnn in algo_configs:
            try:
                agent = make_agent(algo_name, cfg, obs_dim=obs_dim, use_gnn=use_gnn)
                log(f"  [OK] {display_name} created — training...")
                train_agent(agent, cfg, n_episodes=500, warmup_steps=1000, log_interval=100)
                def make_runner(ag=agent):
                    return lambda s: run_episode_agent(ag, cfg, seed=s, deterministic=True)
                algo_runners[display_name] = make_runner()
                log(f"  [OK] {display_name} trained")
            except Exception as e:
                log(f"  [FAIL] {display_name} failed: {e}")
                import traceback
                traceback.print_exc()

        # Run all
        all_runners = {**baselines, **algo_runners}

        for name, runner in all_runners.items():
            try:
                agg = multi_run(runner, n_seeds=n_seeds)
                results_all[topo_name][name] = agg
                log(f"  [OK] {name} evaluated")
            except Exception as e:
                log(f"  [FAIL] {name} evaluation failed: {e}")
                import traceback
                traceback.print_exc()

    # ================ Print Results ================
    for topo_name in configs:
        topo_results = results_all.get(topo_name, {})
        if not topo_results:
            continue

        log(f"\n{'='*100}")
        log(f"  Results for {topo_name}")
        log(f"{'='*100}")

        algo_names = list(topo_results.keys())
        header = f"{'Metric':<25}"
        for name in algo_names:
            header += f" {name:>18}"
        log(header)
        log("-" * (25 + 19 * len(algo_names)))

        for key in display_keys:
            row = f"{key:<25}"
            for name in algo_names:
                val = topo_results[name].get(key, "N/A")
                if isinstance(val, str):
                    row += f" {val:>18}"
                else:
                    row += f" {str(val):>18}"
            log(row)
        log("")

    # ================ Summary Tables ================
    summary_metrics = [
        ("avg_fill_rate", "Fill Rate (higher=better)"),
        ("avg_cost_per_step", "Cost/Step (lower=better)"),
        ("avg_bullwhip_cv", "Bullwhip CV (lower=better)"),
        ("avg_inventory_turnover", "Inv. Turnover (moderate=good)"),
        ("total_holding", "Total Holding Cost"),
        ("total_backlog", "Total Backlog Cost"),
    ]

    for metric_key, metric_label in summary_metrics:
        log(f"\n{'='*100}")
        log(f"  SUMMARY: {metric_label}")
        log(f"{'='*100}")
        header = f"{'Algorithm':<20}"
        for topo in configs:
            header += f" {topo:>18}"
        log(header)
        log("-" * (20 + 19 * len(configs)))

        all_algo_names = set()
        for topo in configs:
            all_algo_names.update(results_all.get(topo, {}).keys())
        all_algo_names = sorted(all_algo_names)

        for name in all_algo_names:
            row = f"{name:<20}"
            for topo in configs:
                topo_res = results_all.get(topo, {}).get(name, {})
                val = topo_res.get(f"{metric_key}_mean", None)
                std = topo_res.get(f"{metric_key}_std", 0)
                if val is not None:
                    row += f" {val:.3f}+/-{std:.3f}".rjust(18)
                else:
                    row += f" {'N/A':>18}"
            log(row)

    log("\n[Done] Comparison complete!")

    # Save to file
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "comparison_all.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
