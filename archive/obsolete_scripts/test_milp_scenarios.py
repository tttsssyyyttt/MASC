"""Comprehensive MILP Oracle test — multiprocessing version.

Covers: 3 topologies × 5 demand patterns × 10 seeds.
Uses full-episode single-solve + multiprocessing for max CPU utilization.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.core.demand import DemandGenerator
from meirp_project.core.network import build_network
from meirp_project.solvers.milp_solver import MILPSolver
from meirp_project.evaluation.metrics import compute_metrics

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


def generate_demand(cfg, pattern, T, n_terms, seed):
    """Generate terminal demand sequence."""
    rng = np.random.default_rng(seed)
    base = cfg.demand_mean

    if pattern == "poisson":
        return np.array([rng.poisson(lam=base, size=n_terms) for _ in range(T)], dtype=np.float64)
    elif pattern == "normal":
        std = getattr(cfg, 'demand_std', base * 0.4)
        return np.maximum(rng.normal(loc=base, scale=std, size=(T, n_terms)), 0).astype(np.float64)
    elif pattern == "seasonal":
        period = 15
        seq = np.zeros((T, n_terms), dtype=np.float64)
        for t in range(T):
            lam = max(1, base / 2 + base / 2 * np.sin(2 * np.pi * t / period))
            seq[t] = rng.poisson(lam=lam, size=n_terms)
        return seq
    elif pattern == "trend":
        start, end = max(1, base / 3), base * 5 / 3
        seq = np.zeros((T, n_terms), dtype=np.float64)
        for t in range(T):
            seq[t] = rng.poisson(lam=start + (end - start) * t / (T - 1), size=n_terms)
        return seq
    elif pattern == "shock":
        seq = np.zeros((T, n_terms), dtype=np.float64)
        for t in range(T):
            lam = base * 3.0 if t == 15 else base
            seq[t] = rng.poisson(lam=lam, size=n_terms)
        return seq
    raise ValueError(f"Unknown pattern: {pattern}")


def _eval_one_seed(args):
    """Evaluate MILP oracle on a single seed. Top-level function for multiprocessing."""
    cfg, pattern, seed = args
    N = cfg.num_nodes
    L = cfg.lead_time
    T = cfg.episode_len
    n_terms = len(build_network(cfg)[3])

    # Generate true demands
    demand_seq = generate_demand(cfg, pattern, T, n_terms, seed)

    # ── Full-episode single solve ──
    solver = MILPSolver(cfg, planning_horizon=T)  # plan the full episode
    orders = solver.solve(
        np.full(N, cfg.init_inventory),
        np.zeros(N),
        np.full((N, L), cfg.init_pipeline),
        demand_seq,
    )

    # Execute the plan (no re-solving!)
    env = MEIRPEnv(cfg)
    obs, _ = env.reset(seed=seed, options={"demand_sequence": demand_seq})
    ih = []; done = False; step = 0
    while not done and step < len(orders):
        # Use heuristic alpha for speed (not LP optimization)
        a = _orders_to_bdq(orders[step], cfg, solver.upstream)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc; ih.append(info); step += 1

    m = compute_metrics(ih, cfg)
    return {
        "cost": m["avg_cost_per_step"],
        "fill": m["avg_fill_rate"],
        "bw": m["avg_bullwhip_cv"],
    }


def _orders_to_bdq(q_vals, cfg, upstream):
    """Convert total q[i] to BDQ format using heuristic alpha."""
    N = cfg.num_nodes
    actions = []
    for i in range(N):
        k = len(upstream[i])
        q = int(q_vals[i])
        if k == 0:
            actions.append(q)
        else:
            # Simple uniform split — fast, good enough for oracle evaluation
            base = q // k
            rem = q % k
            actions.append([base + (1 if j < rem else 0) for j in range(k)])
    return actions


def test_scenario(cfg, pattern, topo_name, n_seeds=10):
    """Test MILP oracle on one scenario (parallel across seeds)."""
    up, _, _, terms = build_network(cfg)
    n_roots = sum(1 for u in up if len(u) == 0)
    supply = n_roots * cfg.max_order

    # Build args list for multiprocessing
    args_list = [(cfg, pattern, seed) for seed in range(n_seeds)]

    # Parallel execution
    n_workers = min(cpu_count(), n_seeds)
    with Pool(n_workers) as pool:
        results = pool.map(_eval_one_seed, args_list)

    costs = [r["cost"] for r in results]
    fills = [r["fill"] for r in results]
    bw_vals = [r["bw"] for r in results]

    # Average demand for ratio
    n_terms_sample = len(terms)
    avg_dem = np.mean([generate_demand(cfg, pattern, 30, n_terms_sample, s).sum(axis=1).mean() for s in range(min(5, n_seeds))])
    ratio = supply / avg_dem if avg_dem > 0 else float("inf")

    return {
        "topo": topo_name,
        "pattern": pattern,
        "supply": supply,
        "avg_demand": round(avg_dem, 1),
        "ratio": round(ratio, 2),
        "cost_mean": round(np.mean(costs), 1),
        "cost_std": round(np.std(costs), 1),
        "fill_mean": round(np.mean(fills), 4),
        "fill_std": round(np.std(fills), 4),
        "fill_min": round(np.min(fills), 4),
        "bw_mean": round(np.mean(bw_vals), 4),
    }


def main():
    n_workers = cpu_count()
    print(f"Using {n_workers} parallel workers\n")

    scenarios = [
        # ── Existing HARD configs ──
        ("env_2layers_hard.yaml", "2layers", "poisson"),
        ("env_3layers_hard.yaml", "3layers", "poisson"),
        ("env_4layers_hard.yaml", "4layers", "poisson"),
        # ── Robustness: all demand patterns on 3-layer ──
        ("env_3layers_hard.yaml", "3layers", "normal"),
        ("env_3layers_hard.yaml", "3layers", "seasonal"),
        ("env_3layers_hard.yaml", "3layers", "trend"),
        ("env_3layers_hard.yaml", "3layers", "shock"),
    ]

    results = []
    for cfg_file, topo, pattern in scenarios:
        cfg = from_yaml(os.path.join(CFG_DIR, cfg_file))
        r = test_scenario(cfg, pattern, topo)
        results.append(r)
        stars = "**" if r["fill_mean"] >= 0.95 else ("*" if r["fill_mean"] >= 0.90 else "")
        print(f"[{topo:8s}][{pattern:<10s}] fill={r['fill_mean']:.4f}±{r['fill_std']:.4f}  cost={r['cost_mean']:.1f}  ratio={r['ratio']}  min={r['fill_min']:.4f} {stars}")

    # ── Summary ──
    print(f"\n{'='*85}")
    print(f"{'Scenario':<22} {'Ratio':>6} {'Cost':>8} {'Fill±Std':>14} {'MinFill':>8}")
    print(f"{'-'*85}")
    for r in results:
        label = f"{r['topo']}/{r['pattern']}"
        print(f"{label:<22} {r['ratio']:>6.2f} {r['cost_mean']:>8.1f} {r['fill_mean']:>8.4f}±{r['fill_std']:.4f} {r['fill_min']:>8.4f}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "milp_oracle_test.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
