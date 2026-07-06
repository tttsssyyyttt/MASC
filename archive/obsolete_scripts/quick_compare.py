"""快速对比实验: 50轮训练 + 基线 + 消融, 汇总所有指标."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.registry import make_agent
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.baselines.ss_policy import SSPolicy
from meirp_project.evaluation.metrics import compute_metrics
from meirp_project.collaboration.sea_wrapper import SEAWrapper
from meirp_project.collaboration.advisor import ExpertAdvisor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "configs")

def train(agent, cfg, n_ep=50, label=""):
    env = MEIRPEnv(cfg)
    rewards = []
    steps = 0
    is_on_policy = hasattr(agent, 'buffer') and hasattr(agent.buffer, 'compute_returns')
    for ep in range(n_ep):
        obs, _ = env.reset()
        ep_r = 0.0; done = False
        while not done:
            actions = agent.get_actions(obs)
            next_obs, r, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            agent.store_transition(obs, actions, r, next_obs, np.full(cfg.num_nodes, float(terminated)))
            steps += 1
            if not is_on_policy and hasattr(agent, 'batch_size') and len(agent.buffer) >= agent.batch_size:
                if steps >= 500: agent.update()
            ep_r += r.sum(); obs = next_obs
        if is_on_policy and len(agent.buffer) > 0:
            agent.update()
        rewards.append(ep_r)
        if (ep+1) % 50 == 0:
            print(f"  [{label}] Ep {ep+1}/{n_ep} | AvgR: {np.mean(rewards[-50:]):.1f}")
    return rewards

def evaluate(agent_or_policy, cfg, n_seeds=3, is_baseline=False):
    all_m = []
    for seed in range(n_seeds):
        env = MEIRPEnv(cfg); obs, _ = env.reset(seed=seed); info_h = []; done = False
        while not done:
            actions = agent_or_policy.get_actions(env) if is_baseline else agent_or_policy.get_actions(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated; info_h.append(info)
        all_m.append(compute_metrics(info_h, cfg))
    result = {}
    for k in ["total_cost","avg_cost_per_step","total_holding","total_backlog",
              "avg_fill_rate","min_fill_rate","avg_stockout_rate","min_stockout_rate",
              "avg_bullwhip_cv","avg_inventory_turnover","holding_ratio","backlog_ratio"]:
        vals = [m.get(k,0) for m in all_m]
        result[k] = np.mean(vals); result[f"{k}_std"] = np.std(vals)
    return result

def print_detail(name, r):
    print(f"  {name}:")
    print(f"    成本/步={r['avg_cost_per_step']:.1f}+/-{r['avg_cost_per_step_std']:.1f}  "
          f"满足率={r['avg_fill_rate']:.4f}  缺货率={r['avg_stockout_rate']:.4f}  "
          f"牛鞭CV={r['avg_bullwhip_cv']:.4f}  周转率={r['avg_inventory_turnover']:.2f}")
    print(f"    库存={r['total_holding']:.1f}  缺货={r['total_backlog']:.1f}  "
          f"最低满足率={r['min_fill_rate']:.4f}")

def run_experiment(topo_key="3layers", n_ep=50, gnn_extra=25, n_seeds=2):
    cfg = from_yaml(os.path.join(CFG_DIR, f"env_{topo_key}.yaml"))
    obs_dim = 3 + 3 * cfg.lead_time
    print(f"\n{'='*90}")
    print(f"  MEIRP 快速对比实验: {topo_key} ({cfg.num_nodes}节点, max_order={cfg.max_order})")
    print(f"  训练={n_ep}轮(GNN+{gnn_extra}), 评估种子={n_seeds}")
    print(f"{'='*90}")

    # ---- 基线 ----
    print("\n--- 基线策略 (无需训练) ---")
    results = {}
    order = []
    for name, policy in [("BaseStock", BaseStockPolicy(cfg)), ("(s,S)", SSPolicy(cfg))]:
        r = evaluate(policy, cfg, n_seeds, is_baseline=True)
        results[name] = r; order.append(name)
        print_detail(name, r)

    # ---- 训练6种算法 ----
    algo_cfgs = [
        ("IQL",       "iql",   False, dict(hidden_dim=128, lr=1e-3, gamma=0.99, buffer_capacity=10000, batch_size=32, epsilon_decay=5000, target_update_freq=50)),
        ("IQL+GNN",   "iql",   True,  dict(hidden_dim=128, lr=5e-4, gamma=0.99, buffer_capacity=100000, batch_size=32, epsilon_decay=8000, target_update_freq=100)),
        ("QMIX",      "qmix",  False, dict(hidden_dim=128, lr=5e-4, gamma=0.99, buffer_capacity=10000, batch_size=32, target_update_freq=50, mixer_hidden=32)),
        ("QMIX+GNN",  "qmix",  True,  dict(hidden_dim=128, lr=3e-4, gamma=0.99, buffer_capacity=100000, batch_size=64, target_update_freq=100, mixer_hidden=32)),
        ("MAPPO",     "mappo", False, dict(hidden_dim=128, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, entropy_coef=0.01)),
        ("MAPPO+GNN", "mappo", True,  dict(hidden_dim=128, lr_actor=2e-4, lr_critic=8e-4, gamma=0.99, entropy_coef=0.02)),
    ]

    for name, algo, use_gnn, kw in algo_cfgs:
        total_ep = n_ep + (gnn_extra if use_gnn else 0)
        print(f"\n--- 训练: {name} ({total_ep}轮) ---")
        agent = make_agent(algo, cfg, obs_dim=obs_dim, use_gnn=use_gnn, **kw)
        t0 = time.time()
        rewards = train(agent, cfg, n_ep=total_ep, label=name)
        elapsed = time.time() - t0
        r = evaluate(agent, cfg, n_seeds)
        results[name] = r; order.append(name)
        print(f"  时间={elapsed:.0f}s | 最终奖励={np.mean(rewards[-min(10,len(rewards)):]):.1f}")
        print_detail(name, r)

    # ---- 汇总对比表 ----
    print(f"\n{'='*90}")
    print("  详细指标对比表")
    print(f"{'='*90}")
    metrics_cfg = [
        ("avg_cost_per_step",      "成本/步",      "lower"),
        ("total_holding",          "库存成本",     "lower"),
        ("total_backlog",          "缺货成本",     "lower"),
        ("avg_stockout_rate",      "缺货率",       "lower"),
        ("avg_fill_rate",          "满足率",       "higher"),
        ("avg_bullwhip_cv",        "牛鞭效应 CV",  "lower"),
        ("avg_inventory_turnover", "库存周转率",   "moderate"),
    ]
    for mk, ml, direction in metrics_cfg:
        arrow = "v" if direction=="lower" else ("^" if direction=="higher" else "~")
        print(f"\n  {ml} ({arrow}):")
        header = f"  {'算法':<14}"
        for n in order: header += f"  {n:>16}"
        print(header)
        print("  " + "-" * (14 + 18 * len(order)))
        vals = [results[n].get(mk, float('nan')) for n in order]
        valid = [v for v in vals if not np.isnan(v)]
        best = min(valid) if direction=="lower" and valid else (max(valid) if direction=="higher" and valid else None)
        row = f"  {'':<14}"
        for i, n in enumerate(order):
            v = results[n].get(mk, float('nan'))
            s = results[n].get(f"{mk}_std", 0)
            mark = " *" if best is not None and not np.isnan(v) and abs(v-best)<1e-9 else "  "
            fmt = ".4f" if isinstance(v, float) and abs(v)<10 else ".1f"
            row += f"  {v:{fmt}}{mark}" if not np.isnan(v) else f"  {'N/A':>16}"
        print(row)

    # ---- 排名 ----
    print(f"\n{'='*90}")
    print("  综合排名 (按满足率排序)")
    print(f"{'='*90}")
    ranked = sorted([(n, results[n]['avg_fill_rate'], results[n]['avg_cost_per_step'],
                       results[n]['avg_bullwhip_cv'], results[n]['avg_stockout_rate']) for n in order],
                    key=lambda x: -x[1])
    for i, (name, fr, cost, bw, so) in enumerate(ranked, 1):
        print(f"  {i}. {name:<14}  满足率={fr:.4f}  成本/步={cost:.1f}  牛鞭CV={bw:.4f}  缺货率={so:.4f}")

    # ---- 消融实验 (IQL) ----
    print(f"\n{'='*90}")
    print("  消融实验: IQL 算法 (MADRL / +GNN / +SEA / +GNN+SEA)")
    print(f"{'='*90}")
    ab_results = {}; ab_order = ["MADRL(IQL)","+GNN","+SEA","+GNN+SEA"]; ab_n_ep = n_ep

    # 1. MADRL only
    print("\n  [1/4] MADRL (IQL)...")
    a1 = make_agent("iql", cfg, obs_dim=obs_dim, use_gnn=False, hidden_dim=128, lr=1e-3, buffer_capacity=10000)
    _ = train(a1, cfg, n_ep=ab_n_ep, label="MADRL")
    ab_results["MADRL(IQL)"] = evaluate(a1, cfg, n_seeds)
    print_detail("MADRL(IQL)", ab_results["MADRL(IQL)"])

    # 2. +GNN
    print("\n  [2/4] +GNN...")
    a2 = make_agent("iql", cfg, obs_dim=obs_dim, use_gnn=True, hidden_dim=128, lr=5e-4, buffer_capacity=100000)
    _ = train(a2, cfg, n_ep=ab_n_ep+gnn_extra, label="+GNN")
    ab_results["+GNN"] = evaluate(a2, cfg, n_seeds)
    print_detail("+GNN", ab_results["+GNN"])

    # 3. +SEA
    print("\n  [3/4] +SEA...")
    a3 = make_agent("iql", cfg, obs_dim=obs_dim, use_gnn=False, hidden_dim=128, lr=1e-3, buffer_capacity=10000)
    _ = train(a3, cfg, n_ep=ab_n_ep, label="+SEA")
    expert = ExpertAdvisor(cfg, intervention_threshold=0.3)
    w3 = SEAWrapper(a3, sea=expert, use_sea=True)
    ab_results["+SEA"] = evaluate(w3, cfg, n_seeds)
    print_detail("+SEA", ab_results["+SEA"])

    # 4. +GNN+SEA
    print("\n  [4/4] +GNN+SEA...")
    a4 = make_agent("iql", cfg, obs_dim=obs_dim, use_gnn=True, hidden_dim=128, lr=5e-4, buffer_capacity=100000)
    _ = train(a4, cfg, n_ep=ab_n_ep+gnn_extra, label="+GNN+SEA")
    expert2 = ExpertAdvisor(cfg, intervention_threshold=0.3)
    w4 = SEAWrapper(a4, sea=expert2, use_sea=True)
    ab_results["+GNN+SEA"] = evaluate(w4, cfg, n_seeds)
    print_detail("+GNN+SEA", ab_results["+GNN+SEA"])

    # ---- 消融对比表 ----
    print(f"\n  消融对比:")
    for mk, ml, direction in metrics_cfg:
        arrow = "v" if direction=="lower" else ("^" if direction=="higher" else "~")
        vals = [ab_results[n].get(mk, float('nan')) for n in ab_order]
        valid = [v for v in vals if not np.isnan(v)]
        best = min(valid) if direction=="lower" and valid else (max(valid) if direction=="higher" and valid else None)
        print(f"  {ml} ({arrow}): ", end="")
        parts = []
        for i, n in enumerate(ab_order):
            v = ab_results[n].get(mk, float('nan'))
            mark = "*" if best is not None and not np.isnan(v) and abs(v-best)<1e-9 else ""
            fmt = ".4f" if (isinstance(v, float) and abs(v)<10) else ".1f"
            parts.append(f"{n}={v:{fmt}}{mark}")
        print(" | ".join(parts))

    # ---- 消融排名 ----
    print(f"\n  消融排名 (按满足率):")
    ab_ranked = sorted([(n, ab_results[n]['avg_fill_rate'], ab_results[n]['avg_cost_per_step'],
                          ab_results[n]['avg_bullwhip_cv']) for n in ab_order],
                       key=lambda x: -x[1])
    for i, (name, fr, cost, bw) in enumerate(ab_ranked, 1):
        print(f"  {i}. {name:<14}  满足率={fr:.4f}  成本/步={cost:.1f}  牛鞭CV={bw:.4f}")

    print(f"\n{'='*90}")
    print("  实验完成!")
    print(f"{'='*90}")
    return results, ab_results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--topo", type=str, default="3layers")
    p.add_argument("--ep", type=int, default=50)
    args = p.parse_args()
    run_experiment(topo_key=args.topo, n_ep=args.ep)
