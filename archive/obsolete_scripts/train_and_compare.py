"""Train all algorithms (with and without GNN) and compare.

Goal: MADRL+GNN > MADRL consistently across metrics.

Prints ALL metrics during evaluation:
- Total cost & cost/step
- Stockout rate (缺货率)
- Bullwhip CV (需求变异放大)
- Inventory turnover rate (库存周转率)
- Fill rate, holding cost, backlog cost breakdown

Usage:
    python meirp_project/scripts/train_and_compare.py
    python meirp_project/scripts/train_and_compare.py --exp meirp_project/configs/experiment.yaml
    python meirp_project/scripts/train_and_compare.py --topologies 2layer 3layer
"""

import sys
import os
import argparse
import time

# Fix Windows console encoding for Chinese characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import yaml
import torch

from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.registry import make_agent
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.baselines.ss_policy import SSPolicy
from meirp_project.evaluation.metrics import compute_metrics


def load_algo_config(config_path, exp_dir=None):
    """Load algorithm config from YAML (resolve relative to experiment dir)."""
    if not os.path.isabs(config_path) and exp_dir is not None:
        config_path = os.path.join(exp_dir, config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_agent_from_config(algo_config, cfg, obs_dim, device):
    """Create agent from algorithm config dict + env config."""
    algo_name = algo_config['algorithm']
    use_gnn = algo_config.get('use_gnn', False)

    kwargs = {}
    for key in ['hidden_dim', 'lr', 'gamma', 'epsilon_start', 'epsilon_end',
                'epsilon_decay', 'buffer_capacity', 'batch_size', 'target_update_freq',
                'encoder_update_freq']:
        if key in algo_config:
            kwargs[key] = algo_config[key]
    if 'mixer_hidden' in algo_config:
        kwargs['mixer_hidden'] = algo_config['mixer_hidden']
    for key in ['lr_actor', 'lr_critic', 'gae_lambda', 'clip_epsilon',
                'epochs_per_update', 'mini_batch_size', 'entropy_coef',
                'value_loss_coef', 'max_grad_norm']:
        if key in algo_config:
            kwargs[key] = algo_config[key]

    return make_agent(algo_name, cfg, obs_dim=obs_dim, use_gnn=use_gnn,
                      device=device, **kwargs)


def train_agent(agent, cfg, n_episodes=300, warmup_steps=500,
                log_interval=50, algo_name="", update_freq=1):
    """Train an agent for n_episodes."""
    env = MEIRPEnv(cfg)
    episode_rewards = []
    step_count = 0
    is_mappo = not hasattr(agent, 'batch_size') and hasattr(agent, 'buffer')

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            actions = agent.get_actions(obs, deterministic=False)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            dones = np.full(cfg.num_nodes, float(terminated))
            agent.store_transition(obs, actions, rewards, next_obs, dones)

            step_count += 1
            if not is_mappo and hasattr(agent, 'batch_size'):
                if len(agent.buffer) >= agent.batch_size and step_count >= warmup_steps:
                    if step_count % update_freq == 0:
                        agent.update()

            ep_reward += rewards.sum()
            obs = next_obs

        if is_mappo and len(agent.buffer) > 0:
            agent.update()

        episode_rewards.append(ep_reward)

        if (ep + 1) % log_interval == 0:
            avg = np.mean(episode_rewards[-log_interval:])
            eps = agent._get_epsilon() if hasattr(agent, '_get_epsilon') else 0.0
            mem = ""
            if torch.cuda.is_available():
                mem = f" | GPU: {torch.cuda.memory_allocated()/1024/1024:.0f}MB"
            print(f"  [{algo_name}] Ep {ep+1}/{n_episodes} | "
                  f"Avg Reward: {avg:.1f} | Eps: {eps:.3f}{mem}")

    return episode_rewards


def evaluate_agent(agent_or_policy, cfg, n_seeds=5, is_baseline=False):
    """Evaluate agent/policy over multiple seeds. Returns detailed metrics."""
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

    # Aggregate
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


def print_eval_detail(name, result, log_fn=print):
    """Print all evaluation metrics for one algorithm."""
    log_fn(f"    ┌──────────────────────────────────────────────────────────")
    log_fn(f"    │ {name}")
    log_fn(f"    ├──────────────────────────────────────────────────────────")
    log_fn(f"    │ 总成本 (Total Cost):        {result['total_cost']:>12.1f} ± {result['total_cost_std']:>8.1f}")
    log_fn(f"    │ 成本/步 (Cost/Step):        {result['avg_cost_per_step']:>12.1f} ± {result['avg_cost_per_step_std']:>8.1f}")
    log_fn(f"    │ 库存成本 (Holding):         {result['total_holding']:>12.1f} ± {result['total_holding_std']:>8.1f}")
    log_fn(f"    │ 缺货成本 (Backlog):         {result['total_backlog']:>12.1f} ± {result['total_backlog_std']:>8.1f}")
    log_fn(f"    │ 缺货率 (Stockout Rate):     {result['avg_stockout_rate']:>12.4f} ± {result['avg_stockout_rate_std']:>8.4f}")
    log_fn(f"    │ 满足率 (Fill Rate):         {result['avg_fill_rate']:>12.4f} ± {result['avg_fill_rate_std']:>8.4f}")
    log_fn(f"    │ 需求变异 (Bullwhip CV):     {result['avg_bullwhip_cv']:>12.4f} ± {result['avg_bullwhip_cv_std']:>8.4f}")
    log_fn(f"    │ 库存周转率 (Inv Turnover):  {result['avg_inventory_turnover']:>12.2f} ± {result['avg_inventory_turnover_std']:>8.2f}")
    log_fn(f"    └──────────────────────────────────────────────────────────")


def print_comparison_table(results, display_order, log_fn=print):
    """Print a side-by-side comparison table of all algorithms."""
    # Key metrics to display
    metrics_config = [
        ("avg_cost_per_step",      "成本/步",      "lower", ".1f"),
        ("total_holding",          "库存成本",     "lower", ".1f"),
        ("total_backlog",          "缺货成本",     "lower", ".1f"),
        ("avg_stockout_rate",      "缺货率",       "lower", ".4f"),
        ("avg_fill_rate",          "满足率",       "higher",".4f"),
        ("avg_bullwhip_cv",        "Bullwhip CV",  "lower", ".4f"),
        ("avg_inventory_turnover", "库存周转率",   "moderate", ".2f"),
    ]

    log_fn("\n" + "=" * 140)
    log_fn("  详细指标对比表")
    log_fn("=" * 140)

    for metric_key, metric_label, direction, fmt in metrics_config:
        log_fn(f"\n  {metric_label} ({'↓' if direction == 'lower' else '↑' if direction == 'higher' else '~'} {'越低越好' if direction == 'lower' else '越高越好' if direction == 'higher' else '适中为好'})")
        header = f"  {'Algorithm':<16}"
        for name in display_order:
            header += f"  {name:>18}"
        log_fn(header)
        log_fn("  " + "-" * (16 + 20 * len(display_order)))

        # Find best value for highlighting
        vals = []
        for name in display_order:
            if name in results:
                vals.append(results[name].get(metric_key, float('nan')))
            else:
                vals.append(float('nan'))
        valid_vals = [v for v in vals if not np.isnan(v)]
        if direction == "lower" and valid_vals:
            best_val = min(valid_vals)
        elif direction == "higher" and valid_vals:
            best_val = max(valid_vals)
        else:
            best_val = None

        line = f"  {'':<16}"
        for idx, name in enumerate(display_order):
            if name in results:
                val = results[name].get(metric_key, float('nan'))
                std = results[name].get(f"{metric_key}_std", 0)
                if np.isnan(val):
                    cell = "N/A"
                else:
                    cell = f"{val:{fmt}}±{std:{fmt}}"
                    if best_val is not None and abs(val - best_val) < 1e-9:
                        cell = f"★{val:{fmt}}±{std:{fmt}}"
                line += f"  {cell:>18}"
            else:
                line += f"  {'N/A':>18}"
        log_fn(line)


def print_compact_summary(results, display_order, log_fn=print):
    """Print a compact summary table: one row per algorithm, key metrics as columns."""
    log_fn("\n" + "=" * 140)
    log_fn("  紧凑汇总表 (Compact Summary)")
    log_fn("=" * 140)

    header = f"  {'Algorithm':<16} {'Cost/Step':>12} {'Stockout':>10} {'FillRate':>10} {'Bullwhip':>10} {'Turnover':>10}"
    log_fn(header)
    log_fn("  " + "-" * len(header))

    for name in display_order:
        if name not in results:
            continue
        r = results[name]
        line = (
            f"  {name:<16} "
            f"{r.get('avg_cost_per_step', 0):>12.1f} "
            f"{r.get('avg_stockout_rate', 0):>10.4f} "
            f"{r.get('avg_fill_rate', 0):>10.4f} "
            f"{r.get('avg_bullwhip_cv', 0):>10.4f} "
            f"{r.get('avg_inventory_turnover', 0):>10.2f}"
        )
        log_fn(line)


def run_single_topology(env_cfg_path, exp_cfg, exp_dir, n_train_episodes,
                        gnn_extra_episodes, n_eval_seeds, log_interval,
                        device, lines):
    """Run training + evaluation for a single topology."""

    cfg = from_yaml(env_cfg_path)
    obs_dim = 3 + 3 * cfg.lead_time

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"\n  Topology: {cfg.num_nodes} nodes, {len(cfg.edges)} edges, "
        f"max_order={cfg.max_order}, lead_time={cfg.lead_time}, "
        f"episode_len={cfg.episode_len}")

    # ===================== Baselines (no training) =====================
    log("\n--- Baselines (no training needed) ---")
    bs = BaseStockPolicy(cfg)
    ss = SSPolicy(cfg)
    bs_result = evaluate_agent(bs, cfg, n_eval_seeds, is_baseline=True)
    ss_result = evaluate_agent(ss, cfg, n_eval_seeds, is_baseline=True)

    print_eval_detail("BaseStock", bs_result, log)
    print_eval_detail("(s,S)", ss_result, log)

    # ===================== Train algorithms from YAML =====================
    algo_entries = exp_cfg.get('algorithms', [])
    trained_agents = {}
    results = {
        "BaseStock": bs_result,
        "(s,S)": ss_result,
    }

    for entry in algo_entries:
        name = entry['name']
        config_path = entry['config']

        log(f"\n--- Training: {name} ---")
        log(f"    Config: {config_path}")

        algo_config = load_algo_config(config_path, exp_dir)
        log(f"    Algorithm: {algo_config['algorithm']}, use_gnn: {algo_config.get('use_gnn', False)}")

        total_steps = n_train_episodes * cfg.episode_len
        is_gnn = algo_config.get('use_gnn', False)
        if is_gnn:
            total_steps += gnn_extra_episodes * cfg.episode_len
        if 'epsilon_decay' not in algo_config or algo_config.get('epsilon_decay', 0) <= 1:
            algo_config['epsilon_decay'] = int(total_steps * 0.5)

        actual_episodes = n_train_episodes + (gnn_extra_episodes if is_gnn else 0)

        agent = create_agent_from_config(algo_config, cfg, obs_dim, device)

        t0 = time.time()
        uf = 8 if is_gnn else 4
        rewards = train_agent(agent, cfg, n_episodes=actual_episodes,
                              log_interval=log_interval, algo_name=name,
                              update_freq=uf)
        elapsed = time.time() - t0
        log(f"    Training time: {elapsed:.1f}s | "
            f"Final avg reward (last 50): {np.mean(rewards[-50:]):.1f}")

        trained_agents[name] = agent

        # Evaluate
        result = evaluate_agent(agent, cfg, n_eval_seeds)
        results[name] = result
        print_eval_detail(name, result, log)

    # ===================== Comparison Table =====================
    display_order = ["BaseStock", "(s,S)"]
    for entry in algo_entries:
        display_order.append(entry['name'])

    print_comparison_table(results, display_order, log)
    print_compact_summary(results, display_order, log)

    return results, display_order


def main():
    parser = argparse.ArgumentParser(description="Train and compare algorithms (BDQ version)")
    parser.add_argument("--exp", type=str, default="meirp_project/configs/experiment.yaml",
                        help="Path to experiment config YAML")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/cpu)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override training episodes")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override eval seeds")
    parser.add_argument("--topologies", nargs='+', type=str, default=None,
                        help="Topology configs to run (e.g. 2layer 3layer). "
                             "If not specified, uses env_config from experiment YAML")
    args = parser.parse_args()

    with open(args.exp, 'r', encoding='utf-8') as f:
        exp_cfg = yaml.safe_load(f)

    exp_dir = os.path.dirname(os.path.abspath(args.exp))

    device = args.device or exp_cfg.get('training', {}).get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = 'cpu'

    train_cfg = exp_cfg.get('training', {})
    n_train_episodes = args.episodes or train_cfg.get('episodes', 500)
    gnn_extra_episodes = train_cfg.get('gnn_extra_episodes', 0)
    n_eval_seeds = args.seeds or exp_cfg.get('evaluation', {}).get('n_seeds', 5)
    log_interval = train_cfg.get('eval_interval', 50)

    # Determine topologies to run
    topo_configs = args.topologies
    if topo_configs is None:
        # Use single env_config from experiment YAML
        env_cfg_path = exp_cfg['env_config']
        if not os.path.isabs(env_cfg_path):
            env_cfg_path = os.path.join(exp_dir, env_cfg_path)
        topo_configs = [env_cfg_path]
    else:
        # Map shorthand to config files
        topo_map = {
            '2layer': 'env_2layers.yaml',
            '3layer': 'env_3layers.yaml',
            '4layer': 'env_4layers.yaml',
        }
        resolved = []
        for t in topo_configs:
            if t in topo_map:
                resolved.append(os.path.join(exp_dir, topo_map[t]))
            elif os.path.exists(t):
                resolved.append(t)
            else:
                print(f"WARNING: Unknown topology '{t}', skipping")
        topo_configs = resolved

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 140)
    log("  BDQ Training & Evaluation — Branching Dueling Q-Network for Supply Chain")
    log("=" * 140)
    log(f"  Device: {device}" +
        (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))
    log(f"  Training episodes: {n_train_episodes}")
    log(f"  GNN extra episodes: {gnn_extra_episodes}")
    log(f"  Eval seeds: {n_eval_seeds}")
    log(f"  Topologies: {len(topo_configs)}")

    all_results = {}

    for topo_path in topo_configs:
        topo_name = os.path.splitext(os.path.basename(topo_path))[0]

        log("\n" + "=" * 140)
        log(f"  Topology: {topo_name}")
        log("=" * 140)

        results, display_order = run_single_topology(
            topo_path, exp_cfg, exp_dir, n_train_episodes,
            gnn_extra_episodes, n_eval_seeds, log_interval,
            device, lines,
        )
        all_results[topo_name] = (results, display_order)

    # ===================== Cross-Topology Summary =====================
    if len(topo_configs) > 1:
        log("\n" + "=" * 140)
        log("  跨拓扑汇总 (Cross-Topology Summary)")
        log("=" * 140)

        # Collect all algorithm names
        all_algo_names = set()
        for results, _ in all_results.values():
            all_algo_names.update(results.keys())
        algo_order = sorted(all_algo_names)

        summary_metrics = [
            ("avg_cost_per_step", "成本/步", ".1f"),
            ("avg_stockout_rate", "缺货率", ".4f"),
            ("avg_fill_rate", "满足率", ".4f"),
            ("avg_bullwhip_cv", "Bullwhip CV", ".4f"),
            ("avg_inventory_turnover", "库存周转率", ".2f"),
        ]

        for metric_key, metric_label, fmt in summary_metrics:
            log(f"\n  {metric_label}")
            header = f"  {'Algorithm':<16}"
            for topo_name in all_results:
                header += f"  {topo_name:>18}"
            log(header)
            log("  " + "-" * len(header))

            for algo in algo_order:
                line = f"  {algo:<16}"
                for topo_name in all_results:
                    results, _ = all_results[topo_name]
                    if algo in results:
                        val = results[algo].get(metric_key, float('nan'))
                        std = results[algo].get(f"{metric_key}_std", 0)
                        if np.isnan(val):
                            line += f"  {'N/A':>18}"
                        else:
                            cell = f"{val:{fmt}}±{std:{fmt}}"
                            line += f"  {cell:>18}"
                    else:
                        line += f"  {'N/A':>18}"
                log(line)

    # ===================== GNN vs No-GNN Head-to-Head =====================
    log("\n" + "=" * 140)
    log("  GNN vs No-GNN 对比")
    log("=" * 140)

    for topo_name, (results, _) in all_results.items():
        algo_entries = exp_cfg.get('algorithms', [])
        base_names = [e['name'] for e in algo_entries if not e.get('config', '').endswith('_gnn.yaml')]

        gnn_pairs = []
        for base_name in base_names:
            gnn_name = base_name + "_gnn"
            if gnn_name in results and base_name in results:
                gnn_pairs.append((gnn_name, base_name))

        if not gnn_pairs:
            continue

        log(f"\n  [{topo_name}]")
        for gnn_name, base_name in gnn_pairs:
            gnn_r = results[gnn_name]
            base_r = results[base_name]

            fr_gnn = gnn_r['avg_fill_rate']
            fr_base = base_r['avg_fill_rate']
            cost_gnn = gnn_r['avg_cost_per_step']
            cost_base = base_r['avg_cost_per_step']
            so_gnn = gnn_r['avg_stockout_rate']
            so_base = base_r['avg_stockout_rate']
            bw_gnn = gnn_r['avg_bullwhip_cv']
            bw_base = base_r['avg_bullwhip_cv']

            wins = sum([
                fr_gnn > fr_base,
                cost_gnn < cost_base,
                so_gnn < so_base,
                bw_gnn < bw_base,
            ])

            log(f"    {gnn_name} vs {base_name}: "
                f"FillRate {fr_gnn:.3f} vs {fr_base:.3f} ({'GNN✓' if fr_gnn > fr_base else 'BASE✓'}), "
                f"Cost {cost_gnn:.1f} vs {cost_base:.1f} ({'GNN✓' if cost_gnn < cost_base else 'BASE✓'}), "
                f"Stockout {so_gnn:.4f} vs {so_base:.4f} ({'GNN✓' if so_gnn < so_base else 'BASE✓'}), "
                f"Bullwhip {bw_gnn:.4f} vs {bw_base:.4f} ({'GNN✓' if bw_gnn < bw_base else 'BASE✓'}) "
                f"→ {wins}/4 GNN wins")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'comparison_bdq.txt')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
