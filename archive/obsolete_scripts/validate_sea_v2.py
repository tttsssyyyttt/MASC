"""SEA validation v2: multi-topology + convergence curves.

Runs Pure RL vs SEA on all 3 topologies with per-episode metrics.
Purpose: show SEA converges faster and helps more on harder topologies.
No reward shaping modifications — let difficulty come from topology complexity.
"""

import sys, os, time, json, warnings
warnings.filterwarnings("ignore")

if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.registry import make_agent
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.evaluation.metrics import compute_metrics
from meirp_project.collaboration.escalation import EscalationController
from meirp_project.collaboration.milp_advisor import MILPAdvisor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "configs")


def evaluate_episode(agent, cfg, seed=0, escalation=None, is_baseline=False):
    """Run one eval episode, return per-step metrics."""
    env = MEIRPEnv(cfg)
    obs, _ = env.reset(seed=seed)
    if escalation: escalation.reset_episode()
    info_h = []
    done = False
    while not done:
        if is_baseline:
            actions = agent.get_actions(env)
        else:
            actions = agent.get_actions(obs, deterministic=True)
        if escalation:
            fr = info_h[-1]["fill_rates"] if info_h else np.ones(cfg.num_nodes)
            mc_fn = agent.estimate_uncertainty if hasattr(agent, 'estimate_uncertainty') else None
            actions, _, _ = escalation.step(
                obs, actions, env.inventory, env.backlog,
                env.pipeline, env.demand, fr,
                mc_uncertainty_fn=mc_fn)
        obs, r, term, trunc, info = env.step(actions)
        done = term or trunc
        info_h.append(info)
    return compute_metrics(info_h, cfg)


def train_with_eval(agent, cfg, n_ep=200, warmup=500, escalation=None, label="", eval_every=10, n_eval_seeds=3):
    """Train RL agent with periodic evaluation. Returns per-ep curves."""
    env = MEIRPEnv(cfg)
    rewards = []
    eval_curve = []  # list of {ep, fill, cost, bullwhip, ...}
    steps = 0

    for ep in range(n_ep):
        obs, _ = env.reset()
        if escalation: escalation.reset_episode()
        ep_r = 0.0; done = False
        while not done:
            actions = agent.get_actions(obs)
            if escalation:
                fr = np.ones(cfg.num_nodes)
                mc_fn = agent.estimate_uncertainty if hasattr(agent, 'estimate_uncertainty') else None
                actions, _, _ = escalation.step(
                    obs, actions, env.inventory, env.backlog,
                    env.pipeline, env.demand, fr,
                    mc_uncertainty_fn=mc_fn)
            nobs, r, term, trunc, info = env.step(actions)
            done = term or trunc
            agent.store_transition(obs, actions, r, nobs, np.full(cfg.num_nodes, float(term)))
            steps += 1
            if hasattr(agent, 'batch_size') and len(agent.buffer) >= agent.batch_size:
                if steps >= warmup: agent.update()
            ep_r += r.sum(); obs = nobs
        rewards.append(ep_r)

        # Periodic evaluation
        if (ep + 1) % eval_every == 0:
            eval_ms = []
            for s in range(n_eval_seeds):
                eval_ms.append(evaluate_episode(agent, cfg, seed=s, escalation=escalation))
            curve = {
                "ep": ep + 1,
                "avg_reward": float(np.mean(rewards[-eval_every:])),
                "avg_fill_rate": float(np.mean([m["avg_fill_rate"] for m in eval_ms])),
                "avg_cost_per_step": float(np.mean([m["avg_cost_per_step"] for m in eval_ms])),
                "avg_bullwhip_cv": float(np.mean([m.get("avg_bullwhip_cv", 0) for m in eval_ms])),
                "avg_stockout_rate": float(np.mean([m["avg_stockout_rate"] for m in eval_ms])),
            }
            eval_curve.append(curve)

        if (ep + 1) % 50 == 0:
            stats_str = ""
            if escalation:
                s = escalation.get_escalation_stats()
                stats_str = (f"| SPC={s['spc_alarm_rate']:.3f} Algo={s['algo_rate']:.3f} "
                             f"Human={s['human_rate']:.3f} OODthr={s.get('ood_threshold',0):.2f}")
            print(f"  [{label}] Ep{ep+1}/{n_ep} R={np.mean(rewards[-50:]):.1f} {stats_str}")

    return rewards, eval_curve


def run_topo(topo_key, n_ep=200, n_seeds=3):
    """Run Pure RL vs SEA on one topology."""
    cfg = from_yaml(os.path.join(CFG_DIR, f"env_{topo_key}.yaml"))
    obs_dim = 3 + 3 * cfg.lead_time

    print(f"\n{'='*70}")
    print(f"  Topology: {topo_key} | {cfg.num_nodes} nodes | "
          f"max_order={cfg.max_order} | B/H terminal={cfg.B[-1]/cfg.H[-1]:.1f}")
    print(f"{'='*70}")

    # --- Baselines ---
    bs = BaseStockPolicy(cfg)
    bs_evals = [evaluate_episode(bs, cfg, seed=s, is_baseline=True) for s in range(n_seeds)]
    bs_fill = np.mean([m["avg_fill_rate"] for m in bs_evals])
    bs_cost = np.mean([m["avg_cost_per_step"] for m in bs_evals])
    bs_bw = np.mean([m.get("avg_bullwhip_cv", 0) for m in bs_evals])
    print(f"  BaseStock baseline: cost={bs_cost:.1f} fill={bs_fill:.4f} bullwhip={bs_bw:.4f}")

    # --- Pure RL ---
    print(f"\n  --- Pure IQL ---")
    agent_pure = make_agent("iql", cfg, obs_dim=obs_dim, device='cuda',
                            hidden_dim=128, lr=1e-3, gamma=0.99,
                            buffer_capacity=10000, batch_size=32,
                            epsilon_decay=5000, target_update_freq=50)
    t0 = time.time()
    _, pure_curve = train_with_eval(agent_pure, cfg, n_ep=n_ep, escalation=None,
                                     label=f"Pure-{topo_key}", eval_every=10, n_eval_seeds=n_seeds)
    pure_time = time.time() - t0

    # --- SEA ---
    print(f"\n  --- SEA-IQL ---")
    agent_sea = make_agent("iql", cfg, obs_dim=obs_dim, device='cuda',
                           hidden_dim=128, lr=1e-3, gamma=0.99,
                           buffer_capacity=10000, batch_size=32,
                           epsilon_decay=5000, target_update_freq=50)
    escalation = EscalationController(cfg, k_spc=3.0)
    t0 = time.time()
    _, sea_curve = train_with_eval(agent_sea, cfg, n_ep=n_ep, escalation=escalation,
                                    label=f"SEA-{topo_key}", eval_every=10, n_eval_seeds=n_seeds)
    sea_time = time.time() - t0

    # --- Final eval ---
    pure_final_evals = [evaluate_episode(agent_pure, cfg, seed=s) for s in range(10)]
    sea_final_evals = [evaluate_episode(agent_sea, cfg, seed=s, escalation=escalation) for s in range(10)]

    # --- Escalation stats ---
    esc_stats = escalation.get_escalation_stats()

    result = {
        "topology": topo_key,
        "n_nodes": cfg.num_nodes,
        "max_order": cfg.max_order,
        "bs_fill": bs_fill, "bs_cost": bs_cost, "bs_bullwhip": bs_bw,
        "pure_final_fill": float(np.mean([m["avg_fill_rate"] for m in pure_final_evals])),
        "pure_final_cost": float(np.mean([m["avg_cost_per_step"] for m in pure_final_evals])),
        "pure_final_bw": float(np.mean([m.get("avg_bullwhip_cv", 0) for m in pure_final_evals])),
        "sea_final_fill": float(np.mean([m["avg_fill_rate"] for m in sea_final_evals])),
        "sea_final_cost": float(np.mean([m["avg_cost_per_step"] for m in sea_final_evals])),
        "sea_final_bw": float(np.mean([m.get("avg_bullwhip_cv", 0) for m in sea_final_evals])),
        "pure_time": pure_time, "sea_time": sea_time,
        "escalation": {"spc_rate": esc_stats["spc_alarm_rate"],
                       "algo_rate": esc_stats["algo_rate"],
                       "human_rate": esc_stats["human_rate"],
                       "total_steps": esc_stats["total_steps"]},
        "pure_curve": pure_curve,
        "sea_curve": sea_curve,
    }

    # Print summary
    print(f"\n  Results for {topo_key}:")
    print(f"    {'':<15} {'Cost/Step':>10} {'Fill':>8} {'Bullwhip':>8}")
    print(f"    {'BaseStock':<15} {bs_cost:>10.1f} {bs_fill:>8.4f} {bs_bw:>8.4f}")
    print(f"    {'Pure IQL':<15} {result['pure_final_cost']:>10.1f} {result['pure_final_fill']:>8.4f} {result['pure_final_bw']:>8.4f}")
    print(f"    {'SEA-IQL':<15} {result['sea_final_cost']:>10.1f} {result['sea_final_fill']:>8.4f} {result['sea_final_bw']:>8.4f}")
    fill_gain = (result['sea_final_fill'] - result['pure_final_fill']) * 100
    cost_gain = (result['pure_final_cost'] - result['sea_final_cost']) / result['pure_final_cost'] * 100
    print(f"    SEA gain: fill +{fill_gain:.1f}pp, cost -{cost_gain:.1f}%")
    print(f"    Escalation: SPC={esc_stats['spc_alarm_rate']:.2%}  "
          f"Algo={esc_stats['algo_rate']:.2%}  Human={esc_stats['human_rate']:.2%}")

    return result


def main():
    n_ep = 200
    all_results = {}

    for topo in ["2layers", "3layers", "4layers"]:
        all_results[topo] = run_topo(topo, n_ep=n_ep, n_seeds=3)

    # --- Cross-topology summary ---
    print(f"\n{'='*70}")
    print(f"  Cross-Topology Summary")
    print(f"{'='*70}")
    print(f"  {'Topo':<10} {'Nodes':<6} {'max_q':<6} {'BS Fill':>8} "
          f"{'Pure Fill':>10} {'SEA Fill':>10} {'Δ Fill':>8} {'Δ Cost%':>8}")
    print(f"  {'-'*70}")
    for topo in ["2layers", "3layers", "4layers"]:
        r = all_results[topo]
        d_fill = (r['sea_final_fill'] - r['pure_final_fill']) * 100
        d_cost = (r['pure_final_cost'] - r['sea_final_cost']) / r['pure_final_cost'] * 100
        print(f"  {topo:<10} {r['n_nodes']:<6} {r['max_order']:<6} "
              f"{r['bs_fill']:>8.4f} {r['pure_final_fill']:>10.4f} "
              f"{r['sea_final_fill']:>10.4f} {d_fill:>+8.2f}pp {d_cost:>+8.1f}%")

    # --- Convergence speed comparison ---
    print(f"\n{'='*70}")
    print(f"  Convergence Speed: Episodes to reach BS fill rate")
    print(f"{'='*70}")
    for topo in ["2layers", "3layers", "4layers"]:
        r = all_results[topo]
        bs_fill = r['bs_fill']
        pure_conv = None; sea_conv = None
        for pt in r['pure_curve']:
            if pt['avg_fill_rate'] >= bs_fill:
                pure_conv = pt['ep']; break
        for pt in r['sea_curve']:
            if pt['avg_fill_rate'] >= bs_fill:
                sea_conv = pt['ep']; break
        speedup = f"{pure_conv/sea_conv:.1f}x" if (pure_conv and sea_conv) else "N/A"
        print(f"  {topo}: Pure→{pure_conv}ep  SEA→{sea_conv}ep  Speedup={speedup}")

    # --- Save curves for plotting ---
    save_path = os.path.join(os.path.dirname(SCRIPT_DIR), "results", "validate_sea_v2.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {save_path}")


if __name__ == "__main__":
    main()
