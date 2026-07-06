"""完整对比实验: 逐节点指标 + 逐层指标 + 基线 + RL算法 + 消融实验.

用法:
    set PYTHONIOENCODING=utf-8 && python meirp_project/scripts/full_experiment.py
    set PYTHONIOENCODING=utf-8 && python meirp_project/scripts/full_experiment.py --topo 3layers --ep 100
"""

import sys, os, time, argparse, warnings
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
from meirp_project.core.network import build_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "configs")


# ============================================================
# 辅助函数
# ============================================================
def train(agent, cfg, n_ep=50, label="", warmup=500):
    """训练智能体"""
    env = MEIRPEnv(cfg); rewards = []; steps = 0
    is_on_policy = hasattr(agent, 'buffer') and hasattr(agent.buffer, 'compute_returns')
    for ep in range(n_ep):
        obs, _ = env.reset(); ep_r = 0.0; done = False
        while not done:
            actions = agent.get_actions(obs)
            nobs, r, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            agent.store_transition(obs, actions, r, nobs, np.full(cfg.num_nodes, float(terminated)))
            steps += 1
            if not is_on_policy and hasattr(agent, 'batch_size') and len(agent.buffer) >= agent.batch_size:
                if steps >= warmup: agent.update()
            ep_r += r.sum(); obs = nobs
        if is_on_policy and len(agent.buffer) > 0:
            agent.update()
        rewards.append(ep_r)
    return rewards


def evaluate(agent_or_policy, cfg, n_seeds=3, is_baseline=False):
    """评估并返回聚合指标 (多种子平均)"""
    all_m = []
    for seed in range(n_seeds):
        env = MEIRPEnv(cfg); obs, _ = env.reset(seed=seed); info_h = []; done = False
        while not done:
            actions = agent_or_policy.get_actions(env) if is_baseline else agent_or_policy.get_actions(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated; info_h.append(info)
        all_m.append(compute_metrics(info_h, cfg))

    # 聚合: 标量用均值+/-std, 数组用均值
    result = {}
    array_keys = ["per_node_cost", "per_node_cost_per_step", "per_node_holding", "per_node_backlog",
                  "per_node_fill_rate", "per_node_stockout_rate", "per_node_bullwhip_cv", "per_node_turnover"]
    scalar_keys = ["total_cost", "avg_cost_per_step", "total_holding", "total_backlog",
                   "avg_fill_rate", "min_fill_rate", "avg_stockout_rate", "min_stockout_rate",
                   "avg_bullwhip_cv", "avg_inventory_turnover", "holding_ratio", "backlog_ratio"]

    for k in scalar_keys:
        vals = [m.get(k, 0) for m in all_m]
        result[k] = np.mean(vals); result[f"{k}_std"] = np.std(vals)

    for k in array_keys:
        arrs = [m[k] for m in all_m if k in m]
        if arrs:
            stacked = np.stack(arrs, axis=0)  # (seeds, N)
            result[k] = stacked.mean(axis=0)  # (N,) mean across seeds
            result[f"{k}_std"] = stacked.std(axis=0)  # (N,) std across seeds

    # 逐层聚合
    if "per_layer" in all_m[0]:
        layer_keys = list(all_m[0]["per_layer"].keys())
        per_layer = {}
        for lk in layer_keys:
            vals = [m["per_layer"].get(lk, 0) for m in all_m]
            per_layer[lk] = np.mean(vals); per_layer[f"{lk}_std"] = np.std(vals)
        result["per_layer"] = per_layer
        result["n_layers"] = all_m[0].get("n_layers", 0)

    return result


# ============================================================
# 打印函数
# ============================================================
def print_sep(title, width=100):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def print_per_node_table(result, cfg, title="逐节点指标"):
    """打印每个节点的详细指标表"""
    n = cfg.num_nodes
    layers = np.array(cfg.layers)
    upstream, _, _, _ = build_network(cfg)

    print(f"\n  --- {title} ---")
    header = (f"  {'节点':<6} {'层':<4} {'上游数':<6} "
              f"{'成本/步':>10} {'满足率':>8} {'缺货率':>8} {'牛鞭CV':>8} {'周转率':>8} "
              f"{'库存成本':>10} {'缺货成本':>10}")
    print(header)
    print("  " + "-" * len(header))

    for i in range(n):
        cost = result.get('per_node_cost_per_step', np.zeros(n))[i]
        fr = result.get('per_node_fill_rate', np.zeros(n))[i]
        so = result.get('per_node_stockout_rate', np.zeros(n))[i]
        bw = result.get('per_node_bullwhip_cv', np.zeros(n))[i]
        turn = result.get('per_node_turnover', np.zeros(n))[i]
        hold = result.get('per_node_holding', np.zeros(n))[i]
        back = result.get('per_node_backlog', np.zeros(n))[i]
        n_up = len(upstream[i])
        node_type = "根" if n_up == 0 else ("终端" if i in set(cfg.layers) else "中间")
        print(f"  {f'节点{i}({node_type})':<6} {layers[i]:<4} {n_up:<6} "
              f"{cost:>10.1f} {fr:>8.4f} {so:>8.4f} {bw:>8.4f} {turn:>8.2f} "
              f"{hold:>10.1f} {back:>10.1f}")


def print_per_layer_table(result, cfg, title="逐层聚合指标"):
    """打印每层的聚合指标表"""
    per_layer = result.get("per_layer", {})
    if not per_layer:
        print("  (无逐层数据)")
        return

    n_layers = result.get("n_layers", 0)
    print(f"\n  --- {title} ---")
    header = (f"  {'层级':<8} {'节点数':<8} "
              f"{'平均成本/步':>12} {'平均满足率':>10} {'平均缺货率':>10} "
              f"{'平均牛鞭CV':>10} {'平均周转率':>10} {'平均库存成本':>12} {'平均缺货成本':>12}")
    print(header)
    print("  " + "-" * len(header))

    layers_arr = np.array(cfg.layers)
    for l in range(n_layers):
        n_nodes = int((layers_arr == l).sum())
        cost = per_layer.get(f"layer_{l}_cost_per_step", 0)
        fr = per_layer.get(f"layer_{l}_fill_rate", 0)
        so = per_layer.get(f"layer_{l}_stockout_rate", 0)
        bw = per_layer.get(f"layer_{l}_bullwhip_cv", 0)
        turn = per_layer.get(f"layer_{l}_turnover", 0)
        hold = per_layer.get(f"layer_{l}_holding", 0)
        back = per_layer.get(f"layer_{l}_backlog", 0)
        print(f"  {'Layer '+str(l):<8} {n_nodes:<8} "
              f"{cost:>12.1f} {fr:>10.4f} {so:>10.4f} "
              f"{bw:>10.4f} {turn:>10.2f} {hold:>12.1f} {back:>12.1f}")


def print_global_comparison(all_results, order, cfg):
    """打印全局对比表 (每个算法的汇总指标)"""
    metrics_cfg = [
        ("avg_cost_per_step",      "成本/步 (avg cost/step)",        "lower"),
        ("total_holding",          "库存成本 (holding cost)",        "lower"),
        ("total_backlog",          "缺货成本 (backlog cost)",        "lower"),
        ("avg_stockout_rate",      "缺货率 (stockout rate)",         "lower"),
        ("avg_fill_rate",          "满足率 (fill rate)",             "higher"),
        ("avg_bullwhip_cv",        "牛鞭效应 CV (bullwhip CV)",      "lower"),
        ("avg_inventory_turnover", "库存周转率 (inventory turnover)","moderate"),
    ]

    print_sep("全局指标对比表")
    for mk, ml, direction in metrics_cfg:
        arrow = { "lower": "v (越低越好)", "higher": "^ (越高越好)", "moderate": "~ (适中为好)" }[direction]
        print(f"\n  {ml} {arrow}:")
        header = f"  {'算法':<14}"
        for n in order: header += f"  {n:>18}"
        print(header); print("  " + "-" * (14 + 20*len(order)))

        vals = [all_results[n].get(mk, float('nan')) for n in order]
        valid = [v for v in vals if not np.isnan(v)]
        if direction == "lower" and valid: best = min(valid)
        elif direction == "higher" and valid: best = max(valid)
        else: best = None

        row = f"  {'':<14}"
        for i, n in enumerate(order):
            v = all_results[n].get(mk, float('nan'))
            s = all_results[n].get(f"{mk}_std", 0)
            mark = " *" if best is not None and not np.isnan(v) and abs(v-best)<1e-9 else "  "
            fmt = ".4f" if (isinstance(v, float) and (abs(v)<10 or mk.endswith('rate') or mk.endswith('cv'))) else ".1f"
            row += f"  {v:{fmt}}{mark}" if not np.isnan(v) else f"  {'N/A':>18}"
        print(row)


def print_per_layer_comparison(all_results, order, cfg):
    """打印逐层对比表"""
    n_layers = max([r.get("n_layers", 0) for r in all_results.values()])
    if n_layers == 0: return

    layer_metrics = [
        ("layer_{l}_cost_per_step", "成本/步"),
        ("layer_{l}_fill_rate",     "满足率"),
        ("layer_{l}_stockout_rate", "缺货率"),
        ("layer_{l}_bullwhip_cv",  "牛鞭CV"),
        ("layer_{l}_turnover",      "周转率"),
    ]

    for l in range(n_layers):
        print_sep(f"Layer {l} 逐层对比")
        layers_arr = np.array(cfg.layers)
        n_nodes = int((layers_arr == l).sum())
        print(f"  Layer {l}: {n_nodes} 个节点")

        for mk_pattern, ml in layer_metrics:
            mk = mk_pattern.format(l=l)
            print(f"\n  {ml}:")
            header = f"  {'算法':<14}"
            for n in order: header += f"  {n:>16}"
            print(header); print("  " + "-" * (14 + 18*len(order)))

            vals = []
            for n in order:
                pl = all_results[n].get("per_layer", {})
                v = pl.get(mk, float('nan'))
                vals.append(v)
            valid = [v for v in vals if not np.isnan(v)]

            row = f"  {'':<14}"
            for i, n in enumerate(order):
                v = vals[i]
                fmt = ".4f" if isinstance(v, float) and abs(v)<10 else ".1f"
                row += f"  {v:{fmt}}" if not np.isnan(v) else f"  {'N/A':>16}"
            print(row)


# ============================================================
# 主实验
# ============================================================
def run_experiment(topo_key="3layers", n_ep=100, gnn_extra=50, n_seeds=3):
    cfg = from_yaml(os.path.join(CFG_DIR, f"env_{topo_key}.yaml"))
    obs_dim = 3 + 3 * cfg.lead_time

    print_sep(f"MEIRP 完整对比实验: {topo_key}")
    print(f"  拓扑: {cfg.num_nodes}节点, {len(cfg.edges)}边, max_order={cfg.max_order}, "
          f"lead_time={cfg.lead_time}, episode_len={cfg.episode_len}")
    print(f"  层级分布: {dict(zip(*np.unique(cfg.layers, return_counts=True)))}")
    print(f"  B/H: {[f'{cfg.B[i]/cfg.H[i]:.1f}' for i in range(cfg.num_nodes)]}")
    print(f"  训练: {n_ep}轮 (GNN额外+{gnn_extra}轮), 评估种子: {n_seeds}")

    all_results = {}
    order = []

    # ======================== 基线 ========================
    print_sep("基线策略 (无需训练)")
    for name, policy in [("BaseStock", BaseStockPolicy(cfg)), ("(s,S)", SSPolicy(cfg))]:
        r = evaluate(policy, cfg, n_seeds, is_baseline=True)
        all_results[name] = r; order.append(name)
        print(f"\n  [{name}] 全局: 成本/步={r['avg_cost_per_step']:.1f}  满足率={r['avg_fill_rate']:.4f}  "
              f"缺货率={r['avg_stockout_rate']:.4f}  牛鞭CV={r['avg_bullwhip_cv']:.4f}  周转率={r['avg_inventory_turnover']:.2f}")
        print_per_node_table(r, cfg, f"{name} 逐节点指标")
        print_per_layer_table(r, cfg, f"{name} 逐层指标")

    # ======================== RL 算法 ========================
    algo_cfgs = [
        ("IQL",       "iql",   False, dict(hidden_dim=128, lr=1e-3, gamma=0.99, buffer_capacity=10000, batch_size=32, epsilon_decay=5000, target_update_freq=50)),
        ("IQL+GNN",   "iql",   True,  dict(hidden_dim=128, lr=3e-4, gamma=0.99, buffer_capacity=200000, batch_size=64, epsilon_decay=20000, target_update_freq=200, encoder_update_freq=16)),
        ("QMIX",      "qmix",  False, dict(hidden_dim=128, lr=5e-4, gamma=0.99, buffer_capacity=10000, batch_size=32, target_update_freq=50, mixer_hidden=32)),
        ("QMIX+GNN",  "qmix",  True,  dict(hidden_dim=128, lr=2e-4, gamma=0.99, buffer_capacity=200000, batch_size=64, target_update_freq=200, mixer_hidden=64, encoder_update_freq=16)),
        ("MAPPO",     "mappo", False, dict(hidden_dim=128, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, entropy_coef=0.01)),
        ("MAPPO+GNN", "mappo", True,  dict(hidden_dim=128, lr_actor=1e-4, lr_critic=5e-4, gamma=0.995, entropy_coef=0.03, epochs_per_update=5)),
    ]

    for name, algo, use_gnn, kw in algo_cfgs:
        print_sep(f"训练: {name} ({n_ep}轮)")
        agent = make_agent(algo, cfg, obs_dim=obs_dim, use_gnn=use_gnn, device='cuda', **kw)
        t0 = time.time()
        rewards = train(agent, cfg, n_ep=n_ep, label=name)
        elapsed = time.time() - t0
        r = evaluate(agent, cfg, n_seeds)
        all_results[name] = r; order.append(name)
        print(f"  时间={elapsed:.0f}s | 最终奖励(最后10轮)={np.mean(rewards[-10:]):.1f}")
        print(f"  全局: 成本/步={r['avg_cost_per_step']:.1f}  满足率={r['avg_fill_rate']:.4f}  "
              f"缺货率={r['avg_stockout_rate']:.4f}  牛鞭CV={r['avg_bullwhip_cv']:.4f}")
        print_per_node_table(r, cfg, f"{name} 逐节点指标")
        print_per_layer_table(r, cfg, f"{name} 逐层指标")

    # ======================== 全局 + 逐层对比 ========================
    print_global_comparison(all_results, order, cfg)
    print_per_layer_comparison(all_results, order, cfg)

    # ======================== 消融实验 (IQL) ========================
    print_sep("消融实验: IQL (MADRL / +GNN / +SEA / +GNN+SEA)")
    ab_results = {}; ab_order = ["MADRL(IQL)", "+GNN", "+SEA", "+GNN+SEA"]
    ab_n_ep = n_ep

    ab_configs = [
        ("MADRL(IQL)", False, False),
        ("+GNN",       True,  False),
        ("+SEA",       False, True),
        ("+GNN+SEA",   True,  True),
    ]

    for name, use_gnn, use_sea in ab_configs:
        print(f"\n  [{name}] 训练 {n_ep} 轮...")
        agent = make_agent("iql", cfg, obs_dim=obs_dim, device='cuda', use_gnn=use_gnn,
                           hidden_dim=128, lr=3e-4 if use_gnn else 1e-3,
                           buffer_capacity=200000 if use_gnn else 10000,
                           epsilon_decay=20000 if use_gnn else 5000,
                           batch_size=64 if use_gnn else 32,
                           encoder_update_freq=16 if use_gnn else 4)
        _ = train(agent, cfg, n_ep=n_ep, label=name)

        if use_sea:
            expert = ExpertAdvisor(cfg, intervention_threshold=0.3)
            agent = SEAWrapper(agent, sea=expert, use_sea=True)

        r = evaluate(agent, cfg, n_seeds)
        ab_results[name] = r
        print(f"    成本/步={r['avg_cost_per_step']:.1f}  满足率={r['avg_fill_rate']:.4f}  "
              f"缺货率={r['avg_stockout_rate']:.4f}  牛鞭CV={r['avg_bullwhip_cv']:.4f}")
        print_per_node_table(r, cfg, f"{name} 逐节点")
        print_per_layer_table(r, cfg, f"{name} 逐层")

    print_global_comparison(ab_results, ab_order, cfg)
    print_per_layer_comparison(ab_results, ab_order, cfg)

    # ======================== 最终排名 ========================
    print_sep("最终排名 (按满足率)")
    ranked = sorted([(n, all_results[n]['avg_fill_rate'], all_results[n]['avg_cost_per_step'],
                       all_results[n]['avg_bullwhip_cv'], all_results[n]['avg_stockout_rate']) for n in order],
                    key=lambda x: -x[1])
    for i, (name, fr, cost, bw, so) in enumerate(ranked, 1):
        print(f"  {i:>2}. {name:<14}  满足率={fr:.4f}  成本/步={cost:.1f}  牛鞭CV={bw:.4f}  缺货率={so:.4f}")

    print_sep("消融排名 (按满足率)")
    ab_ranked = sorted([(n, ab_results[n]['avg_fill_rate'], ab_results[n]['avg_cost_per_step'],
                          ab_results[n]['avg_bullwhip_cv']) for n in ab_order], key=lambda x: -x[1])
    for i, (name, fr, cost, bw) in enumerate(ab_ranked, 1):
        print(f"  {i:>2}. {name:<14}  满足率={fr:.4f}  成本/步={cost:.1f}  牛鞭CV={bw:.4f}")

    print(f"\n{'='*100}")
    print("  实验完成!")
    print(f"{'='*100}")

    return all_results, ab_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="完整对比实验 (逐节点+逐层指标)")
    parser.add_argument("--topo", type=str, default="3layers", choices=["2layers","3layers","4layers"])
    parser.add_argument("--ep", type=int, default=100, help="训练轮数")
    parser.add_argument("--gnn-extra", type=int, default=0, help="GNN额外训练轮数 (默认0, 公平对比)")
    parser.add_argument("--seeds", type=int, default=3, help="评估种子数")
    args = parser.parse_args()
    run_experiment(topo_key=args.topo, n_ep=args.ep, gnn_extra=args.gnn_extra, n_seeds=args.seeds)
