"""IQL hyperparameter tuning — focus on reducing stockout rate & cost.

3 configurations:
  - baseline:  default IQL settings
  - proactive: higher gamma, slower eps decay → forward-looking
  - large:     bigger network, larger batch → more capacity

Usage:
    python meirp_project/scripts/tune_iql.py --episodes 800 --seeds 5 --device cuda
    python meirp_project/scripts/tune_iql.py --episodes 100 --seeds 2 --device cuda --configs baseline
"""

import sys
import os
import argparse
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch

from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.iql.agent import IQLAgent
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.baselines.ss_policy import SSPolicy
from meirp_project.evaluation.metrics import compute_metrics


# ==================== IQL Configurations ====================
CONFIGS = {
    "baseline": {
        "hidden_dim": 128,
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 5000,
        "buffer_capacity": 50000,
        "batch_size": 64,
        "target_update_freq": 200,
        "update_freq": 4,
    },
    "proactive": {
        # Higher gamma → value future more → order proactively
        # Slower epsilon decay → explore more
        "hidden_dim": 128,
        "lr": 5e-4,
        "gamma": 0.997,
        "epsilon_start": 1.0,
        "epsilon_end": 0.02,
        "epsilon_decay": 15000,
        "buffer_capacity": 80000,
        "batch_size": 64,
        "target_update_freq": 300,
        "update_freq": 4,
    },
    "large": {
        # Bigger network → more capacity
        "hidden_dim": 256,
        "lr": 5e-4,
        "gamma": 0.995,
        "epsilon_start": 1.0,
        "epsilon_end": 0.03,
        "epsilon_decay": 10000,
        "buffer_capacity": 80000,
        "batch_size": 64,
        "target_update_freq": 200,
        "update_freq": 4,
    },
}


def train_iql(cfg, obs_dim, device, config, n_episodes, log_interval=50, name="iql"):
    """Train a single IQL agent with given config."""
    agent = IQLAgent(
        cfg=cfg, obs_dim=obs_dim, device=device,
        use_gnn=False,
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
        target_update_freq=config["target_update_freq"],
    )

    env = MEIRPEnv(cfg)
    episode_rewards = []
    step_count = 0
    update_freq = config["update_freq"]
    warmup_steps = 500

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_stockout = 0.0
        done = False

        while not done:
            actions = agent.get_actions(obs, deterministic=False)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            dones = np.full(cfg.num_nodes, float(terminated))
            agent.store_transition(obs, actions, rewards, next_obs, dones)

            step_count += 1
            if len(agent.buffer) >= agent.batch_size and step_count >= warmup_steps:
                if step_count % update_freq == 0:
                    agent.update()

            ep_reward += rewards.sum()
            ep_stockout += info.get("backlog_costs", np.zeros(cfg.num_nodes)).sum()
            obs = next_obs

        episode_rewards.append(ep_reward)

        if (ep + 1) % log_interval == 0:
            avg_r = np.mean(episode_rewards[-log_interval:])
            avg_so = ep_stockout / max(cfg.episode_len, 1)
            eps = agent._get_epsilon()
            print(f"  [{name}] Ep {ep+1}/{n_episodes} | "
                  f"AvgR: {avg_r:.1f} | BacklogCost/ep: {avg_so:.0f} | Eps: {eps:.3f}")

    return agent, episode_rewards


def evaluate(agent_or_policy, cfg, n_seeds=5, is_baseline=False):
    """Evaluate and return full metrics dict."""
    all_metrics = []
    for seed in range(n_seeds):
        env = MEIRPEnv(cfg)
        obs, _ = env.reset(seed=seed)
        info_history = []
        done = False

        while not done:
            if is_baseline:
                actions = agent_or_policy.get_actions(env)
            else:
                actions = agent_or_policy.get_actions(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            info_history.append(info)

        metrics = compute_metrics(info_history, cfg)
        all_metrics.append(metrics)

    result = {}
    scalar_keys = [
        "total_cost", "avg_cost_per_step", "total_holding", "total_backlog",
        "avg_fill_rate", "min_fill_rate", "avg_stockout_rate", "min_stockout_rate",
        "avg_bullwhip_cv", "avg_inventory_turnover",
        "holding_ratio", "backlog_ratio",
    ]
    for k in scalar_keys:
        vals = [m.get(k, 0.0) for m in all_metrics]
        result[k] = np.mean(vals)
        result[f"{k}_std"] = np.std(vals)

    return result


def print_result(name, r):
    """Pretty-print metrics for one configuration."""
    print(f"    ┌────────────────────────────────────────────────────")
    print(f"    │ {name}")
    print(f"    ├────────────────────────────────────────────────────")
    print(f"    │ 总成本:          {r['total_cost']:>10.1f} ± {r['total_cost_std']:>8.1f}")
    print(f"    │ 成本/步:         {r['avg_cost_per_step']:>10.1f} ± {r['avg_cost_per_step_std']:>8.1f}")
    print(f"    │ 库存成本:        {r['total_holding']:>10.1f} ± {r['total_holding_std']:>8.1f}")
    print(f"    │ 缺货成本:        {r['total_backlog']:>10.1f} ± {r['total_backlog_std']:>8.1f}")
    print(f"    │ 缺货率:          {r['avg_stockout_rate']:>10.4f} ± {r['avg_stockout_rate_std']:>8.4f}")
    print(f"    │ 满足率:          {r['avg_fill_rate']:>10.4f} ± {r['avg_fill_rate_std']:>8.4f}")
    print(f"    │ Bullwhip CV:     {r['avg_bullwhip_cv']:>10.4f} ± {r['avg_bullwhip_cv_std']:>8.4f}")
    print(f"    │ 库存周转率:      {r['avg_inventory_turnover']:>10.2f} ± {r['avg_inventory_turnover_std']:>8.2f}")
    print(f"    └────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="IQL tuning for stockout reduction")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--configs", nargs='+', type=str, default=None,
                        help="Which configs to run (baseline proactive large)")
    parser.add_argument("--env", type=str,
                        default="meirp_project/configs/env_3layers.yaml")
    args = parser.parse_args()

    cfg = from_yaml(args.env)
    obs_dim = 3 + 3 * cfg.lead_time

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    config_names = args.configs or list(CONFIGS.keys())

    print("=" * 100)
    print("  IQL Tuning — 降低缺货率 & 缺货成本")
    print("=" * 100)
    print(f"  Env: {os.path.basename(args.env)} | Nodes: {cfg.num_nodes} | "
          f"max_order: {cfg.max_order} | lead_time: {cfg.lead_time}")
    print(f"  Device: {args.device}" +
          (f" ({torch.cuda.get_device_name(0)})" if args.device == 'cuda' else ""))
    print(f"  Episodes: {args.episodes} | Eval seeds: {args.seeds}")
    print(f"  Configs: {config_names}")

    # ===================== Baselines =====================
    print("\n--- Baselines ---")
    bs = BaseStockPolicy(cfg)
    ss = SSPolicy(cfg)
    bs_result = evaluate(bs, cfg, args.seeds, is_baseline=True)
    ss_result = evaluate(ss, cfg, args.seeds, is_baseline=True)
    print_result("BaseStock", bs_result)
    print_result("(s,S)", ss_result)

    all_results = {
        "BaseStock": bs_result,
        "(s,S)": ss_result,
    }

    # ===================== Train each config =====================
    for name in config_names:
        if name not in CONFIGS:
            print(f"\n  WARNING: Unknown config '{name}', skipping")
            continue

        config = CONFIGS[name].copy()
        print(f"\n{'='*60}")
        print(f"  Training: IQL-{name}")
        print(f"  Config: {config}")
        print(f"{'='*60}")

        # Auto-compute epsilon_decay if needed
        total_steps = args.episodes * cfg.episode_len
        if config["epsilon_decay"] <= 1:
            config["epsilon_decay"] = int(total_steps * 0.5)

        t0 = time.time()
        agent, rewards = train_iql(
            cfg, obs_dim, args.device, config,
            n_episodes=args.episodes,
            log_interval=max(args.episodes // 10, 10),
            name=name,
        )
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s | "
              f"Final avg reward (last 50): {np.mean(rewards[-50:]):.1f}")

        # Evaluate
        result = evaluate(agent, cfg, args.seeds)
        all_results[f"IQL-{name}"] = result
        print_result(f"IQL-{name}", result)

    # ===================== Comparison =====================
    display_order = ["BaseStock", "(s,S)"] + [f"IQL-{n}" for n in config_names if n in CONFIGS]

    print("\n" + "=" * 100)
    print("  汇总对比")
    print("=" * 100)

    metrics_display = [
        ("avg_cost_per_step", "成本/步", ".1f", "lower"),
        ("total_backlog", "缺货成本", ".1f", "lower"),
        ("avg_stockout_rate", "缺货率", ".4f", "lower"),
        ("avg_fill_rate", "满足率", ".4f", "higher"),
        ("avg_bullwhip_cv", "Bullwhip CV", ".4f", "lower"),
        ("avg_inventory_turnover", "库存周转率", ".2f", "moderate"),
    ]

    for metric_key, metric_label, fmt, direction in metrics_display:
        arrow = "↓" if direction == "lower" else "↑" if direction == "higher" else "~"
        print(f"\n  {metric_label} ({arrow})")

        vals = []
        for n in display_order:
            if n in all_results:
                vals.append(all_results[n].get(metric_key, float('nan')))
        valid = [v for v in vals if not np.isnan(v)]
        best = None
        if direction == "lower" and valid:
            best = min(valid)
        elif direction == "higher" and valid:
            best = max(valid)

        for n in display_order:
            if n not in all_results:
                continue
            r = all_results[n]
            val = r.get(metric_key, float('nan'))
            std = r.get(f"{metric_key}_std", 0)
            if np.isnan(val):
                cell = "N/A"
            else:
                marker = " ★" if best is not None and abs(val - best) < 1e-9 else ""
                cell = f"{val:{fmt}}±{std:{fmt}}{marker}"
            print(f"    {n:<16} {cell}")


if __name__ == "__main__":
    main()
