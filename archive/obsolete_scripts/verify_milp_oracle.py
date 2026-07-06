"""Quick verification: MILP with TRUE demand as oracle upper bound (v2 — padded horizon)."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.core.demand import DemandGenerator
from meirp_project.collaboration.milp_advisor import MILPAdvisor
from meirp_project.evaluation.metrics import compute_metrics

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")

for topo in ["2layers", "3layers", "4layers"]:
    cfg = from_yaml(os.path.join(CFG_DIR, f"env_{topo}_hard.yaml"))

    from meirp_project.core.network import build_network
    up, down, _, terms = build_network(cfg)
    n_roots = sum(1 for u in up if len(u) == 0)
    supply = n_roots * cfg.max_order
    demand_total = len(terms) * cfg.demand_mean
    ratio = supply / demand_total if demand_total > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"  {topo}: {cfg.num_nodes} nodes | roots={n_roots} terms={len(terms)} | supply={supply} demand={demand_total:.0f} ratio={ratio:.2f}")

    HORIZON = 30
    advisor = MILPAdvisor(cfg, planning_horizon=HORIZON)
    all_costs, all_fills = [], []

    for seed in range(5):
        env = MEIRPEnv(cfg)
        rng = np.random.default_rng(seed)
        dgen = DemandGenerator(cfg, rng)
        T = cfg.episode_len
        # Pre-generate TRUE demands + pad beyond episode to avoid end-of-horizon effects
        raw_seq = np.array([dgen.generate(len(env.terminals)) for _ in range(T)], dtype=np.float64)
        pad = np.full((HORIZON, len(env.terminals)), cfg.demand_mean, dtype=np.float64)
        demand_seq = np.concatenate([raw_seq, pad], axis=0)

        obs, _ = env.reset(seed=seed, options={"demand_sequence": raw_seq})
        ih = []; done = False; step = 0

        while not done:
            future = demand_seq[step:step + HORIZON]
            a = advisor.get_action(env.inventory, env.backlog, env.pipeline, demand_forecast=future)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc; ih.append(info); step += 1

        m = compute_metrics(ih, cfg)
        all_costs.append(m["avg_cost_per_step"])
        all_fills.append(m["avg_fill_rate"])

    print(f"  MILP-oracle: cost={np.mean(all_costs):.1f}±{np.std(all_costs):.0f}  fill={np.mean(all_fills):.4f}±{np.std(all_fills):.4f}")
    for s in range(5):
        print(f"    seed={s}: cost={all_costs[s]:.1f} fill={all_fills[s]:.4f}")
