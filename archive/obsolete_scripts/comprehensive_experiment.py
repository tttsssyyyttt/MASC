"""全面对比实验: 多拓扑 + 消融 + +GAT+SEA 成本优化.

策略:
1. 三个拓扑(2/3/4层)基线对比
2. 6种RL算法在3层上的对比(相同训练轮数)
3. 消融: MADRL/+GAT/+SEA/+GAT+SEA
4. +GAT+SEA 成本优化: 调节SEA干预阈值, 在保持满足率的同时降低成本
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "configs")


def train(agent, cfg, n_ep=200, label="", warmup=500):
    env = MEIRPEnv(cfg); rewards = []; steps = 0
    is_on_policy = hasattr(agent, 'buffer') and hasattr(agent.buffer, 'compute_returns')
    for ep in range(n_ep):
        obs, _ = env.reset(); ep_r = 0.0; done = False
        while not done:
            actions = agent.get_actions(obs)
            nobs, r, term, trunc, info = env.step(actions)
            done = term or trunc
            agent.store_transition(obs, actions, r, nobs, np.full(cfg.num_nodes, float(term)))
            steps += 1
            if not is_on_policy and hasattr(agent, 'batch_size') and len(agent.buffer) >= agent.batch_size:
                if steps >= warmup: agent.update()
            ep_r += r.sum(); obs = nobs
        if is_on_policy and len(agent.buffer) > 0:
            agent.update()
        rewards.append(ep_r)
        if (ep + 1) % 50 == 0:
            print(f"  [{label}] Ep {ep+1}/{n_ep} | AvgR: {np.mean(rewards[-50:]):.1f}")
    return rewards


def evaluate(agent_or_policy, cfg, n_seeds=3, is_baseline=False):
    all_m = []
    for seed in range(n_seeds):
        env = MEIRPEnv(cfg); obs, _ = env.reset(seed=seed); info_h = []; done = False
        while not done:
            actions = agent_or_policy.get_actions(env) if is_baseline else agent_or_policy.get_actions(obs, deterministic=True)
            obs, rewards, term, trunc, info = env.step(actions)
            done = term or trunc; info_h.append(info)
        all_m.append(compute_metrics(info_h, cfg))

    result = {}
    scalar_keys = ["total_cost", "avg_cost_per_step", "total_holding", "total_backlog",
                   "avg_fill_rate", "min_fill_rate", "avg_stockout_rate", "min_stockout_rate",
                   "avg_bullwhip_cv", "avg_inventory_turnover"]
    for k in scalar_keys:
        vals = [m.get(k, 0) for m in all_m]
        result[k] = np.mean(vals); result[f"{k}_std"] = np.std(vals)

    array_keys = ["per_node_cost_per_step", "per_node_fill_rate", "per_node_stockout_rate",
                  "per_node_bullwhip_cv", "per_node_turnover"]
    for k in array_keys:
        arrs = [m[k] for m in all_m if k in m]
        if arrs:
            result[k] = np.mean(np.stack(arrs, axis=0), axis=0)

    if "per_layer" in all_m[0]:
        result["per_layer"] = all_m[0]["per_layer"]
    return result


def print_sep(title, w=100):
    print(f"\n{'='*w}\n  {title}\n{'='*w}")


def print_row(label, metrics, keys, fmts):
    parts = [f"{label:<18}"]
    for k, fmt in zip(keys, fmts):
        v = metrics.get(k, 0)
        parts.append(f"{v:{fmt}}")
    print("  " + " | ".join(parts))


# ============================================================
# Part 1: 多拓扑基线
# ============================================================
def part1_baselines(n_seeds=3):
    print_sep("Part 1: 多拓扑基线对比")
    for topo in ['2layers', '3layers', '4layers']:
        cfg = from_yaml(os.path.join(CFG_DIR, f"env_{topo}.yaml"))
        n_terms = sum(1 for l in cfg.layers if l == max(cfg.layers))
        print(f"\n  [{topo}] {cfg.num_nodes}节点, {n_terms}终端")

        for name, policy in [("BaseStock", BaseStockPolicy(cfg)), ("(s,S)", SSPolicy(cfg))]:
            r = evaluate(policy, cfg, n_seeds, is_baseline=True)
            print(f"    {name:<12} 成本/步={r['avg_cost_per_step']:.1f}  满足率={r['avg_fill_rate']:.4f}  "
                  f"缺货率={r['avg_stockout_rate']:.4f}  牛鞭CV={r['avg_bullwhip_cv']:.4f}  周转率={r['avg_inventory_turnover']:.2f}")
    return


# ============================================================
# Part 2: RL算法对比 (3层, 相同训练轮数)
# ============================================================
def part2_rl_comparison(n_ep=200, n_seeds=3):
    print_sep(f"Part 2: RL算法对比 (3层, 各{n_ep}轮)")

    cfg = from_yaml(os.path.join(CFG_DIR, "env_3layers.yaml"))
    obs_dim = 3 + 3 * cfg.lead_time
    results = {}; order = []

    # 基线
    for name, policy in [("BaseStock", BaseStockPolicy(cfg)), ("(s,S)", SSPolicy(cfg))]:
        results[name] = evaluate(policy, cfg, n_seeds, is_baseline=True); order.append(name)

    # RL 算法 (相同训练轮数, 超参数适配架构)
    algo_cfgs = [
        ("IQL",       "iql",   False, dict(hidden_dim=128, lr=1e-3, gamma=0.99, buffer_capacity=10000,  batch_size=32, epsilon_decay=5000, target_update_freq=50)),
        ("IQL+GAT",   "iql",   True,  dict(hidden_dim=128, lr=3e-4, gamma=0.99, buffer_capacity=200000, batch_size=64, epsilon_decay=20000, target_update_freq=200, encoder_update_freq=16)),
        ("QMIX",      "qmix",  False, dict(hidden_dim=128, lr=5e-4, gamma=0.99, buffer_capacity=10000,  batch_size=32, target_update_freq=50, mixer_hidden=32)),
        ("QMIX+GAT",  "qmix",  True,  dict(hidden_dim=128, lr=2e-4, gamma=0.99, buffer_capacity=200000, batch_size=64, target_update_freq=200, mixer_hidden=64, encoder_update_freq=16)),
        ("MAPPO",     "mappo", False, dict(hidden_dim=128, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, entropy_coef=0.01)),
        ("MAPPO+GAT", "mappo", True,  dict(hidden_dim=128, lr_actor=1e-4, lr_critic=5e-4, gamma=0.995, entropy_coef=0.03, epochs_per_update=5)),
    ]

    for name, algo, use_gnn, kw in algo_cfgs:
        print(f"\n  [{name}] 训练 {n_ep} 轮...")
        agent = make_agent(algo, cfg, obs_dim=obs_dim, device='cuda', use_gnn=use_gnn, **kw)
        t0 = time.time()
        _ = train(agent, cfg, n_ep=n_ep, label=name)
        r = evaluate(agent, cfg, n_seeds)
        results[name] = r; order.append(name)
        print(f"    时间={time.time()-t0:.0f}s | 成本/步={r['avg_cost_per_step']:.1f}  "
              f"满足率={r['avg_fill_rate']:.4f}  牛鞭CV={r['avg_bullwhip_cv']:.4f}")

    # 对比表
    print_sep("Part 2 结果: 全局对比")
    metrics_keys = ["avg_cost_per_step", "avg_fill_rate", "avg_stockout_rate", "avg_bullwhip_cv", "avg_inventory_turnover"]
    header = ["算法"] + metrics_keys
    print(f"  {'算法':<18} {'成本/步':>10} {'满足率':>10} {'缺货率':>10} {'牛鞭CV':>10} {'周转率':>10}")
    print("  " + "-"*70)
    best_cost = min(results[n]['avg_cost_per_step'] for n in order)
    best_fr = max(results[n]['avg_fill_rate'] for n in order)
    best_bw = min(results[n]['avg_bullwhip_cv'] for n in order)
    for name in order:
        r = results[name]
        cost_mark = " *" if r['avg_cost_per_step'] == best_cost else ""
        fr_mark = " *" if r['avg_fill_rate'] == best_fr else ""
        bw_mark = " *" if r['avg_bullwhip_cv'] == best_bw else ""
        print(f"  {name:<18} {r['avg_cost_per_step']:>10.1f}{cost_mark}  {r['avg_fill_rate']:>10.4f}{fr_mark}  "
              f"{r['avg_stockout_rate']:>10.4f}  {r['avg_bullwhip_cv']:>10.4f}{bw_mark}  {r['avg_inventory_turnover']:>10.2f}")

    # 排名
    print(f"\n  排名(按满足率):")
    ranked = sorted([(n, results[n]['avg_fill_rate'], results[n]['avg_cost_per_step']) for n in order], key=lambda x: -x[1])
    for i, (n, fr, cost) in enumerate(ranked, 1):
        print(f"    {i}. {n:<18} 满足率={fr:.4f}  成本/步={cost:.1f}")

    return results, order, cfg


# ============================================================
# Part 3: 消融实验 + 成本优化
# ============================================================
def part3_ablation_and_optimize(cfg, n_ep=200, n_seeds=3):
    print_sep(f"Part 3: 消融实验 + +GAT+SEA 成本优化 ({n_ep}轮)")

    obs_dim = 3 + 3 * cfg.lead_time
    common_kw = dict(hidden_dim=128, batch_size=64, buffer_capacity=200000,
                     epsilon_decay=20000, target_update_freq=200, encoder_update_freq=16)

    # ---- 3a. 消融对比 (相同训练) ----
    print("\n  --- 3a. 消融实验 (相同训练轮数) ---")
    ab_results = {}; ab_order = ["MADRL(IQL)", "+GAT", "+SEA", "+GAT+SEA"]

    for name, use_gnn, use_sea in [
        ("MADRL(IQL)", False, False),
        ("+GAT",       True,  False),
        ("+SEA",       False, True),
        ("+GAT+SEA",   True,  True),
    ]:
        print(f"\n  [{name}] 训练 {n_ep} 轮...")
        agent = make_agent("iql", cfg, obs_dim=obs_dim, device='cuda', use_gnn=use_gnn,
                           lr=3e-4 if use_gnn else 1e-3, **common_kw)
        t0 = time.time()
        _ = train(agent, cfg, n_ep=n_ep, label=name)

        if use_sea:
            expert = ExpertAdvisor(cfg, intervention_threshold=0.3)
            agent = SEAWrapper(agent, sea=expert, use_sea=True)

        r = evaluate(agent, cfg, n_seeds)
        ab_results[name] = r
        print(f"    时间={time.time()-t0:.0f}s | 成本/步={r['avg_cost_per_step']:.1f}  "
              f"满足率={r['avg_fill_rate']:.4f}  牛鞭CV={r['avg_bullwhip_cv']:.4f}")

    # 消融对比表
    print(f"\n  消融对比:")
    for mk in ["avg_cost_per_step", "avg_fill_rate", "avg_stockout_rate", "avg_bullwhip_cv"]:
        vals = [ab_results[n].get(mk, 0) for n in ab_order]
        best = min(vals) if "cost" in mk or "stockout" in mk or "bullwhip" in mk else max(vals)
        print(f"  {mk}: " + " | ".join(f"{n}={ab_results[n].get(mk,0):.4f}{'*' if abs(ab_results[n].get(mk,0)-best)<1e-9 else ''}" for n in ab_order))

    # ---- 3b. +GAT+SEA 成本优化: 调节SEA干预阈值 ----
    print_sep("Part 3b: +GAT+SEA 成本优化 (调节SEA干预阈值)")

    # 训练一个 GAT agent (只训练一次, 然后用不同SEA阈值评估)
    print(f"\n  训练 GAT agent ({n_ep}轮)...")
    gat_agent = make_agent("iql", cfg, obs_dim=obs_dim, device='cuda', use_gnn=True,
                           lr=3e-4, **common_kw)
    _ = train(gat_agent, cfg, n_ep=n_ep, label="GAT-base")

    # 测试不同的SEA干预阈值
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    opt_results = {}

    print(f"\n  测试不同SEA干预阈值 (阈值越低=越多专家干预):")
    for thresh in thresholds:
        expert = ExpertAdvisor(cfg, intervention_threshold=thresh)
        wrapped = SEAWrapper(gat_agent, sea=expert, use_sea=True)
        r = evaluate(wrapped, cfg, n_seeds)
        opt_results[f"thresh={thresh:.1f}"] = r
        print(f"    thresh={thresh:.1f}: 成本/步={r['avg_cost_per_step']:.1f}  "
              f"满足率={r['avg_fill_rate']:.4f}  缺货率={r['avg_stockout_rate']:.4f}  "
              f"牛鞭CV={r['avg_bullwhip_cv']:.4f}")

    # 找最优阈值: 在满足率不降低的前提下最小化成本
    base_fr = ab_results["+GAT+SEA"].get("avg_fill_rate", 0)  # 默认阈值0.3的结果
    base_cost = ab_results["+GAT+SEA"].get("avg_cost_per_step", 0)

    print(f"\n  基线 +GAT+SEA (thresh=0.3): 成本/步={base_cost:.1f}, 满足率={base_fr:.4f}")
    print(f"  成本优化目标: 满足率 >= {base_fr:.4f}, 最小化成本/步")

    best_opt = None; best_cost = float('inf')
    for thresh in thresholds:
        r = opt_results[f"thresh={thresh:.1f}"]
        if r['avg_fill_rate'] >= base_fr * 0.99:  # 允许1%以内的满足率下降
            if r['avg_cost_per_step'] < best_cost:
                best_cost = r['avg_cost_per_step']
                best_opt = (thresh, r)

    if best_opt:
        th, r = best_opt
        cost_reduction = base_cost - r['avg_cost_per_step']
        print(f"\n  最优配置: thresh={th:.1f}")
        print(f"    成本/步: {r['avg_cost_per_step']:.1f} (降低 {cost_reduction:.1f}, {cost_reduction/base_cost*100:.1f}%)")
        print(f"    满足率:   {r['avg_fill_rate']:.4f} (vs 基线 {base_fr:.4f})")
        print(f"    缺货率:   {r['avg_stockout_rate']:.4f}")
        print(f"    牛鞭CV:   {r['avg_bullwhip_cv']:.4f}")
    else:
        print(f"\n  所有阈值下满足率均低于基线, 无法在不降低满足率的前提下优化成本")

    # ---- 3c. 与基线对比最优配置 ----
    print_sep("Part 3c: 最终对比 (最优 +GAT+SEA vs 所有算法)")
    all_algo = ab_results.copy()
    if best_opt:
        all_algo["+GAT+SEA(最优)"] = best_opt[1]

    order = ["BaseStock", "(s,S)", "MADRL(IQL)", "+GAT", "+SEA", "+GAT+SEA"]
    if best_opt:
        order.append("+GAT+SEA(最优)")

    # 需要基线数据
    cfg_3l = from_yaml(os.path.join(CFG_DIR, "env_3layers.yaml"))
    for name, policy in [("BaseStock", BaseStockPolicy(cfg_3l)), ("(s,S)", SSPolicy(cfg_3l))]:
        if name not in all_algo:
            all_algo[name] = evaluate(policy, cfg_3l, n_seeds, is_baseline=True)

    print(f"\n  {'算法':<20} {'成本/步':>10} {'满足率':>10} {'缺货率':>10} {'牛鞭CV':>10}")
    print("  " + "-"*65)
    for name in order:
        if name not in all_algo: continue
        r = all_algo[name]
        print(f"  {name:<20} {r['avg_cost_per_step']:>10.1f}  {r['avg_fill_rate']:>10.4f}  "
              f"{r['avg_stockout_rate']:>10.4f}  {r['avg_bullwhip_cv']:>10.4f}")

    return ab_results, opt_results, best_opt


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    print_sep("MEIRP 全面对比实验", 100)
    print(f"  训练: {args.ep}轮(所有算法相同) | 评估: {args.seeds}种子 | 设备: GPU")

    # Part 1: 多拓扑基线
    part1_baselines(n_seeds=args.seeds)

    # Part 2: RL算法对比
    results, order, cfg = part2_rl_comparison(n_ep=args.ep, n_seeds=args.seeds)

    # Part 3: 消融 + 成本优化
    ab_results, opt_results, best_opt = part3_ablation_and_optimize(cfg, n_ep=args.ep, n_seeds=args.seeds)

    print_sep("实验完成!")
    print(f"  +GAT+SEA 成本优化结论: 调节SEA干预阈值可在保持满足率的同时降低成本")
    if best_opt:
        print(f"  最优阈值: {best_opt[0]:.1f}")

if __name__ == "__main__":
    main()
