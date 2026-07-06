"""完整对比实验：训练所有算法 + 基线 + 消融实验，汇总所有指标。

用法：
    set PYTHONIOENCODING=utf-8 && python meirp_project/scripts/run_full_comparison.py
    set PYTHONIOENCODING=utf-8 && python meirp_project/scripts/run_full_comparison.py --episodes 300 --seeds 5
"""

import sys
import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import yaml

from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.registry import make_agent
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.baselines.ss_policy import SSPolicy
from meirp_project.evaluation.metrics import compute_metrics
from meirp_project.collaboration.sea_wrapper import SEAWrapper
from meirp_project.collaboration.advisor import ExpertAdvisor
from meirp_project.evaluation.ablation import AblationRunner


# ============================================================
# 训练函数
# ============================================================
def train_agent(agent, cfg, n_episodes=200, warmup_steps=500, log_interval=50, label=""):
    """训练智能体 n_episodes 轮"""
    env = MEIRPEnv(cfg)
    rewards_history = []
    step_count = 0
    is_on_policy = hasattr(agent, 'buffer') and hasattr(agent.buffer, 'compute_returns')

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

            if not is_on_policy:
                if hasattr(agent, 'batch_size') and len(agent.buffer) >= agent.batch_size:
                    if step_count >= warmup_steps:
                        agent.update()

            ep_reward += rewards.sum()
            obs = next_obs

        if is_on_policy and len(agent.buffer) > 0:
            agent.update()

        rewards_history.append(ep_reward)

        if (ep + 1) % log_interval == 0:
            avg = np.mean(rewards_history[-log_interval:])
            print(f"  [{label}] Ep {ep+1}/{n_episodes} | Avg Reward: {avg:.1f}")

    return rewards_history


# ============================================================
# 评估函数
# ============================================================
def evaluate(agent_or_policy, cfg, n_seeds=5, is_baseline=False):
    """评估智能体/策略，返回汇总指标"""
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


# ============================================================
# 打印函数
# ============================================================
def print_header(title):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")


def print_detail(name, result):
    print(f"  ┌─ {name}")
    print(f"  ├─ 总成本 (Total Cost):        {result['total_cost']:>12.1f} +/- {result['total_cost_std']:>8.1f}")
    print(f"  ├─ 成本/步 (Cost/Step):        {result['avg_cost_per_step']:>12.1f} +/- {result['avg_cost_per_step_std']:>8.1f}")
    print(f"  ├─ 库存成本 (Holding):         {result['total_holding']:>12.1f} +/- {result['total_holding_std']:>8.1f}")
    print(f"  ├─ 缺货成本 (Backlog):         {result['total_backlog']:>12.1f} +/- {result['total_backlog_std']:>8.1f}")
    print(f"  ├─ 缺货率 (Stockout Rate):     {result['avg_stockout_rate']:>12.4f} +/- {result['avg_stockout_rate_std']:>8.4f}")
    print(f"  ├─ 满足率 (Fill Rate):         {result['avg_fill_rate']:>12.4f} +/- {result['avg_fill_rate_std']:>8.4f}")
    print(f"  ├─ 牛鞭效应 (Bullwhip CV):     {result['avg_bullwhip_cv']:>12.4f} +/- {result['avg_bullwhip_cv_std']:>8.4f}")
    print(f"  └─ 库存周转率 (Inv Turnover):  {result['avg_inventory_turnover']:>12.2f} +/- {result['avg_inventory_turnover_std']:>8.2f}")


def print_comparison_table(results, order):
    metrics_cfg = [
        ("avg_cost_per_step",      "成本/步",        "lower", ".1f"),
        ("total_holding",          "库存成本",       "lower", ".1f"),
        ("total_backlog",          "缺货成本",       "lower", ".1f"),
        ("avg_stockout_rate",      "缺货率",         "lower", ".4f"),
        ("avg_fill_rate",          "满足率",         "higher",".4f"),
        ("avg_bullwhip_cv",        "牛鞭效应 CV",    "lower", ".4f"),
        ("avg_inventory_turnover", "库存周转率",     "moderate", ".2f"),
    ]

    print_header("详细指标对比表")

    for metric_key, metric_label, direction, fmt in metrics_cfg:
        direction_symbol = "↓" if direction == "lower" else ("↑" if direction == "higher" else "~")
        direction_text = "越低越好" if direction == "lower" else ("越高越好" if direction == "higher" else "适中为好")
        print(f"\n  {metric_label} ({direction_symbol} {direction_text})")

        header = f"  {'算法':<16}"
        for name in order:
            header += f"  {name:>18}"
        print(header)
        print("  " + "-" * (16 + 20 * len(order)))

        vals_list = []
        for name in order:
            if name in results:
                vals_list.append(results[name].get(metric_key, float('nan')))
            else:
                vals_list.append(float('nan'))

        valid = [v for v in vals_list if not np.isnan(v)]
        if direction == "lower" and valid:
            best = min(valid)
        elif direction == "higher" and valid:
            best = max(valid)
        else:
            best = None

        row = f"  {'':<16}"
        for idx, name in enumerate(order):
            if name in results:
                val = results[name].get(metric_key, float('nan'))
                std = results[name].get(f"{metric_key}_std", 0)
                if np.isnan(val):
                    cell = "N/A"
                else:
                    cell = f"{val:{fmt}}+/-{std:{fmt}}"
                if best is not None and not np.isnan(val) and abs(val - best) < 1e-9:
                    cell = f"★{cell}"
                row += f"  {cell:>18}"
            else:
                row += f"  {'N/A':>18}"
        print(row)


def print_compact_summary(results, order):
    print_header("紧凑汇总表")
    header = f"  {'算法':<16} {'成本/步':>12} {'缺货率':>10} {'满足率':>10} {'牛鞭CV':>10} {'周转率':>10}"
    print(header)
    print("  " + "-" * len(header))
    for name in order:
        if name not in results:
            continue
        r = results[name]
        print(f"  {name:<16} "
              f"{r.get('avg_cost_per_step', 0):>12.1f} "
              f"{r.get('avg_stockout_rate', 0):>10.4f} "
              f"{r.get('avg_fill_rate', 0):>10.4f} "
              f"{r.get('avg_bullwhip_cv', 0):>10.4f} "
              f"{r.get('avg_inventory_turnover', 0):>10.2f}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="完整对比实验")
    parser.add_argument("--episodes", type=int, default=200, help="训练轮数")
    parser.add_argument("--gnn-extra", type=int, default=100, help="GNN 额外训练轮数")
    parser.add_argument("--seeds", type=int, default=3, help="评估种子数")
    parser.add_argument("--topology", type=str, default="3layers", choices=["2layer", "3layer", "4layer", "2layers", "3layers", "4layers"])
    args = parser.parse_args()

    # 确定项目根目录和配置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # meirp_project/
    config_dir = os.path.join(project_root, "configs")

    # 拓扑名称: 统一添加 's' (文件名是 env_2layers.yaml 格式)
    topo_name = args.topology.rstrip('s')
    if not topo_name.endswith('layer'):
        topo_name = topo_name + 'layer'
    topo_name = topo_name + 's'
    env_path = os.path.join(config_dir, f"env_{topo_name}.yaml")
    if not os.path.exists(env_path):
        print(f"错误: 找不到配置文件 {env_path}")
        sys.exit(1)

    cfg = from_yaml(env_path)
    obs_dim = 3 + 3 * cfg.lead_time

    print_header(f"MEIRP 完整对比实验")
    print(f"  拓扑: {args.topology} ({cfg.num_nodes}节点, {len(cfg.edges)}边, "
          f"max_order={cfg.max_order}, lead_time={cfg.lead_time})")
    print(f"  训练: {args.episodes}轮 (GNN额外+{args.gnn_extra}轮)")
    print(f"  评估: {args.seeds}个种子")

    # ======================== 基线 (无需训练) ========================
    print_header("基线策略 (无需训练)")
    bs = BaseStockPolicy(cfg)
    ss = SSPolicy(cfg)
    bs_result = evaluate(bs, cfg, args.seeds, is_baseline=True)
    ss_result = evaluate(ss, cfg, args.seeds, is_baseline=True)
    print_detail("BaseStock (order-up-to S)", bs_result)
    print_detail("(s,S) Policy", ss_result)

    # ======================== 训练所有算法 ========================
    all_results = {"BaseStock": bs_result, "(s,S)": ss_result}
    display_order = ["BaseStock", "(s,S)"]

    algo_configs = [
        ("IQL",       "iql",   False, 128, 1e-3, 0.99, 5000, 10000, 32, 50),
        ("IQL+GNN",   "iql",   True,  128, 5e-4, 0.99, 10000, 100000, 32, 100),
        ("QMIX",      "qmix",  False, 128, 5e-4, 0.99, 5000, 10000, 32, 50),
        ("QMIX+GNN",  "qmix",  True,  128, 3e-4, 0.99, 10000, 100000, 64, 100),
        ("MAPPO",     "mappo", False, 128, 3e-4, 0.99, 0, 0, 32, 0),
        ("MAPPO+GNN", "mappo", True,  128, 2e-4, 0.99, 0, 0, 32, 0),
    ]

    for name, algo, use_gnn, hid, lr, gamma, eps_decay, buf_cap, batch, target_freq in algo_configs:
        print_header(f"训练: {name}")
        display_order.append(name)

        total_ep = args.episodes + (args.gnn_extra if use_gnn else 0)
        kw = dict(hidden_dim=hid, lr=lr, gamma=gamma, buffer_capacity=buf_cap,
                  batch_size=batch, target_update_freq=target_freq)

        if algo == "mappo":
            kw = dict(hidden_dim=hid, lr_actor=lr, lr_critic=lr*3, gamma=gamma,
                      entropy_coef=0.02 if use_gnn else 0.01)

        if eps_decay > 0:
            kw['epsilon_decay'] = eps_decay

        agent = make_agent(algo, cfg, obs_dim=obs_dim, use_gnn=use_gnn, **kw)

        t0 = time.time()
        rewards = train_agent(agent, cfg, n_episodes=total_ep, label=name)
        elapsed = time.time() - t0
        print(f"  训练时间: {elapsed:.1f}s | 最终平均奖励(最后50轮): {np.mean(rewards[-50:]):.1f}")

        result = evaluate(agent, cfg, args.seeds)
        all_results[name] = result
        print_detail(name, result)

    # ======================== 对比表 ========================
    print_comparison_table(all_results, display_order)
    print_compact_summary(all_results, display_order)

    # ======================== 消融实验 ========================
    print_header("消融实验 (Ablation): IQL 算法")

    ablation_results = {}
    ablation_order = ["MADRL (IQL)", "+GNN", "+SEA", "+GNN+SEA"]

    # 获取算法配置
    algo_cfg_path = os.path.join(config_dir, "algo_iql.yaml")
    algo_cfg_gnn_path = os.path.join(config_dir, "algo_iql_gnn.yaml")

    if os.path.exists(algo_cfg_path):
        with open(algo_cfg_path, 'r') as f:
            iql_cfg = yaml.safe_load(f)
    else:
        iql_cfg = dict(algorithm='iql', use_gnn=False, hidden_dim=128, lr=1e-3,
                       gamma=0.99, buffer_capacity=10000, batch_size=32)

    # 1. MADRL only (IQL, no GNN)
    print("\n  [1/4] MADRL only (IQL, 无GNN)...")
    agent_madrl = make_agent("iql", cfg, obs_dim=obs_dim, use_gnn=False,
                              hidden_dim=128, lr=1e-3, buffer_capacity=10000)
    _ = train_agent(agent_madrl, cfg, n_episodes=args.episodes, log_interval=100, label="MADRL")
    ablation_results["MADRL (IQL)"] = evaluate(agent_madrl, cfg, args.seeds)

    # 2. +GNN
    print("\n  [2/4] +GNN (IQL + GNN编码器)...")
    agent_gnn = make_agent("iql", cfg, obs_dim=obs_dim, use_gnn=True,
                            hidden_dim=128, lr=5e-4, buffer_capacity=100000)
    _ = train_agent(agent_gnn, cfg, n_episodes=args.episodes+args.gnn_extra, log_interval=100, label="+GNN")
    ablation_results["+GNN"] = evaluate(agent_gnn, cfg, args.seeds)

    # 3. +SEA
    print("\n  [3/4] +SEA (IQL + 专家增强)...")
    agent_sea = make_agent("iql", cfg, obs_dim=obs_dim, use_gnn=False,
                            hidden_dim=128, lr=1e-3, buffer_capacity=10000)
    _ = train_agent(agent_sea, cfg, n_episodes=args.episodes, log_interval=100, label="+SEA")
    expert = ExpertAdvisor(cfg, intervention_threshold=0.3)
    wrapped_sea = SEAWrapper(agent_sea, sea=expert, use_sea=True)
    ablation_results["+SEA"] = evaluate(wrapped_sea, cfg, args.seeds)

    # 4. +GNN+SEA
    print("\n  [4/4] +GNN+SEA (IQL + GNN + 专家增强)...")
    agent_gnn_sea = make_agent("iql", cfg, obs_dim=obs_dim, use_gnn=True,
                                hidden_dim=128, lr=5e-4, buffer_capacity=100000)
    _ = train_agent(agent_gnn_sea, cfg, n_episodes=args.episodes+args.gnn_extra, log_interval=100, label="+GNN+SEA")
    expert2 = ExpertAdvisor(cfg, intervention_threshold=0.3)
    wrapped_gnn_sea = SEAWrapper(agent_gnn_sea, sea=expert2, use_sea=True)
    ablation_results["+GNN+SEA"] = evaluate(wrapped_gnn_sea, cfg, args.seeds)

    # 消融对比表
    print_comparison_table(ablation_results, ablation_order)
    print_compact_summary(ablation_results, ablation_order)

    # ======================== 最终汇总 ========================
    print_header("最终汇总")
    print(f"\n  测试拓扑: {args.topology} ({cfg.num_nodes}节点)")
    print(f"  训练轮数: {args.episodes} (GNN +{args.gnn_extra})")
    print(f"  评估种子: {args.seeds}")
    print(f"\n  算法排名 (按满足率 Fill Rate):")

    all_order = display_order
    ranked = sorted(
        [(n, all_results[n].get('avg_fill_rate', 0)) for n in all_order if n in all_results],
        key=lambda x: -x[1]
    )
    for rank, (name, fr) in enumerate(ranked, 1):
        r = all_results[name]
        print(f"    {rank}. {name:<16}  满足率={fr:.4f}  成本/步={r['avg_cost_per_step']:.1f}  "
              f"牛鞭CV={r['avg_bullwhip_cv']:.4f}  缺货率={r['avg_stockout_rate']:.4f}")

    print(f"\n  消融排名 (按满足率 Fill Rate):")
    ab_ranked = sorted(
        [(n, ablation_results[n].get('avg_fill_rate', 0)) for n in ablation_order if n in ablation_results],
        key=lambda x: -x[1]
    )
    for rank, (name, fr) in enumerate(ab_ranked, 1):
        r = ablation_results[name]
        print(f"    {rank}. {name:<16}  满足率={fr:.4f}  成本/步={r['avg_cost_per_step']:.1f}  "
              f"牛鞭CV={r['avg_bullwhip_cv']:.4f}  缺货率={r['avg_stockout_rate']:.4f}")

    print(f"\n{'='*100}")
    print("  实验完成!")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
