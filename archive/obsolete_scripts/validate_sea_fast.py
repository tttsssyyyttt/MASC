"""SEA validation v3: Big model + SPC→MILP expert + parallel-friendly.

Key changes:
  - hidden_dim=256, batch_size=256 for better GPU utilization
  - SPC alarm → MILP expert directly (dropping too-weak BaseStock)
  - No MC Dropout (unstable with current network size)
  - --topo arg for parallel execution across topologies

Usage (parallel):
  start /b python validate_sea_fast.py --topo 2layers
  start /b python validate_sea_fast.py --topo 3layers
  start /b python validate_sea_fast.py --topo 4layers
"""

import sys, os, time, json, argparse, warnings
warnings.filterwarnings("ignore")
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.registry import make_agent
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.evaluation.metrics import compute_metrics
from meirp_project.collaboration.escalation import EscalationController
from meirp_project.collaboration.milp_advisor import MILPAdvisor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "configs")


# ============================================================
# Simple SPC→MILP escalation (no BaseStock, no MC Dropout)
# ============================================================
class SimpleEscalation:
    """Two-level: RL autonomous | SPC alarm → MILP expert."""

    def __init__(self, cfg, k_spc=3.0):
        from meirp_project.core.health_monitor import HealthMonitor
        self.health = HealthMonitor(cfg.num_nodes)
        self.k_spc = k_spc
        self.milp = MILPAdvisor(cfg)
        self.total_steps = 0
        self.alarms = 0
        self.human_steps = 0

    def step(self, obs, rl_actions, env):
        """Returns (final_actions, level, log)."""
        self.total_steps += 1
        H_t = self.health.update(env.backlog, env.demand, np.ones(env.n))
        in_control = abs(H_t) <= self.k_spc

        if not in_control and self.health.is_warm:
            self.alarms += 1
            try:
                actions = self.milp.get_action(env.inventory, env.backlog, env.pipeline)
                self.human_steps += 1
                return actions, "expert", {"H_t": H_t}
            except Exception:
                pass
        return rl_actions, "rl", {"H_t": H_t}

    def stats(self):
        t = max(self.total_steps, 1)
        return {"spc_alarm": self.alarms/t, "expert": self.human_steps/t,
                "rl": (t-self.alarms)/t, "total": t}


def eval_ep(agent, cfg, seed, escalation=None, is_baseline=False):
    env = MEIRPEnv(cfg); obs, _ = env.reset(seed=seed); ih = []; done = False
    while not done:
        if is_baseline: a = agent.get_actions(env)
        else: a = agent.get_actions(obs, deterministic=True)
        if escalation and not is_baseline: a, _, _ = escalation.step(obs, a, env)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc; ih.append(info)
    return compute_metrics(ih, cfg)


def train(agent, cfg, escalation, n_ep, warmup, label):
    env = MEIRPEnv(cfg); rewards = []; steps = 0; eval_curve = []

    for ep in range(n_ep):
        obs, _ = env.reset(); ep_r = 0.0; done = False
        while not done:
            a = agent.get_actions(obs)
            if escalation: a, _, _ = escalation.step(obs, a, env)
            nobs, r, term, trunc, info = env.step(a)
            done = term or trunc
            agent.store_transition(obs, a, r, nobs, np.full(cfg.num_nodes, float(term)))
            steps += 1
            if len(agent.buffer) >= agent.batch_size and steps >= warmup: agent.update()
            ep_r += r.sum(); obs = nobs
        rewards.append(ep_r)

        if (ep+1) % 10 == 0:
            ms = [eval_ep(agent, cfg, s, escalation) for s in range(3)]
            eval_curve.append({"ep": ep+1, "fill": np.mean([m["avg_fill_rate"] for m in ms]),
                               "cost": np.mean([m["avg_cost_per_step"] for m in ms]),
                               "bw": np.mean([m.get("avg_bullwhip_cv",0) for m in ms])})

        if (ep+1) % 50 == 0:
            if escalation:
                s = escalation.stats()
                extra = f"SPC={s['spc_alarm']:.3f} Exp={s['expert']:.3f}"
            else:
                extra = ""
            print(f"  [{label}] Ep{ep+1}/{n_ep} R={np.mean(rewards[-50:]):.1f} {extra}")

    return rewards, eval_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topo", type=str, required=True, choices=["2layers","3layers","4layers"])
    parser.add_argument("--ep", type=int, default=200)
    args = parser.parse_args()

    cfg = from_yaml(os.path.join(CFG_DIR, f"env_{args.topo}.yaml"))
    obs_dim = 3 + 3 * cfg.lead_time

    print(f"\n{'='*60}\n  {args.topo}: {cfg.num_nodes} nodes, max_order={cfg.max_order}\n{'='*60}")

    # Baselines
    bs = BaseStockPolicy(cfg)
    bs_ms = [eval_ep(bs, cfg, s, is_baseline=True) for s in range(5)]
    bs_fill = np.mean([m["avg_fill_rate"] for m in bs_ms])
    bs_cost = np.mean([m["avg_cost_per_step"] for m in bs_ms])
    print(f"  BaseStock: cost={bs_cost:.1f} fill={bs_fill:.4f}")

    # --- Pure IQL (big model) ---
    print("\n  --- Pure IQL ---")
    pure = make_agent("iql", cfg, obs_dim=obs_dim, device='cuda',
                      hidden_dim=256, lr=1e-3, gamma=0.99,
                      buffer_capacity=50000, batch_size=256,
                      epsilon_decay=8000, target_update_freq=100)
    t0 = time.time()
    r_pure, c_pure = train(pure, cfg, None, args.ep, 1000, f"Pure-{args.topo}")
    pt = time.time() - t0
    pure_ms = [eval_ep(pure, cfg, s) for s in range(10)]
    p_fill = np.mean([m["avg_fill_rate"] for m in pure_ms])
    p_cost = np.mean([m["avg_cost_per_step"] for m in pure_ms])
    p_bw = np.mean([m.get("avg_bullwhip_cv",0) for m in pure_ms])

    # --- SEA-IQL (SPC→MILP) ---
    print("\n  --- SEA-IQL ---")
    sea = make_agent("iql", cfg, obs_dim=obs_dim, device='cuda',
                     hidden_dim=256, lr=1e-3, gamma=0.99,
                     buffer_capacity=50000, batch_size=256,
                     epsilon_decay=8000, target_update_freq=100)
    esc = SimpleEscalation(cfg)
    t0 = time.time()
    r_sea, c_sea = train(sea, cfg, esc, args.ep, 1000, f"SEA-{args.topo}")
    st = time.time() - t0
    sea_ms = [eval_ep(sea, cfg, s, esc) for s in range(10)]
    s_fill = np.mean([m["avg_fill_rate"] for m in sea_ms])
    s_cost = np.mean([m["avg_cost_per_step"] for m in sea_ms])
    s_bw = np.mean([m.get("avg_bullwhip_cv",0) for m in sea_ms])
    s_stats = esc.stats()

    # --- Summary ---
    d_fill = (s_fill - p_fill) * 100
    d_cost = (p_cost - s_cost) / p_cost * 100
    print(f"\n  Results for {args.topo}:")
    print(f"    BaseStock:  cost={bs_cost:.1f} fill={bs_fill:.4f}")
    print(f"    Pure IQL:   cost={p_cost:.1f} fill={p_fill:.4f} bw={p_bw:.4f} time={pt:.0f}s")
    print(f"    SEA-IQL:    cost={s_cost:.1f} fill={s_fill:.4f} bw={s_bw:.4f} time={st:.0f}s")
    print(f"    SEA gain:   fill {d_fill:+.1f}pp  cost {d_cost:+.1f}%")
    print(f"    Escalation: SPC={s_stats['spc_alarm']:.2%} Expert={s_stats['expert']:.2%}")

    # Save
    result = {"topo": args.topo, "bs_fill": bs_fill, "bs_cost": bs_cost,
              "p_fill": p_fill, "p_cost": p_cost, "p_bw": p_bw,
              "s_fill": s_fill, "s_cost": s_cost, "s_bw": s_bw,
              "p_time": pt, "s_time": st, "escalation": s_stats,
              "pure_curve": c_pure, "sea_curve": c_sea}
    out = os.path.join(os.path.dirname(SCRIPT_DIR), "results", f"sea_fast_{args.topo}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f: json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
