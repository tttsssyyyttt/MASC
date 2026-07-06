"""Run MADRL + SEA comparison across all experimental scenarios in parallel.

Launches independent subprocesses for each (topology, demand_pattern) pair.
Each subprocess runs full RL training + evaluation for all algorithms.

Usage:
    python meirp_project/scripts/run_all_scenarios.py --ep 200 --seeds 5
"""
import sys, os, subprocess, json, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FULL_COMPARE = Path(__file__).resolve().parent / "full_compare.py"

SCENARIOS = [
    # (topo_flag, config_file, label)
    ("2layers", "env_2layers_hard.yaml", "2L-Poisson"),
    ("3layers", "env_3layers_hard.yaml", "3L-Poisson"),
    ("4layers", "env_4layers_hard.yaml", "4L-Poisson"),
    ("3layers", "env_3layers_normal.yaml", "3L-Normal"),
    ("3layers", "env_3layers_seasonal.yaml", "3L-Seasonal"),
    ("3layers", "env_3layers_trend.yaml", "3L-Trend"),
    ("3layers", "env_3layers_shock.yaml", "3L-Shock"),
]


def run_one_scenario(args_tuple):
    """Run full comparison for one scenario. Top-level for multiprocessing."""
    topo, config_file, label, episodes, seeds = args_tuple

    config_path = ROOT / "meirp_project" / "configs" / config_file
    if not config_path.exists():
        return {"label": label, "error": f"Config not found: {config_path}"}

    cmd = [
        sys.executable, str(FULL_COMPARE),
        "--topo", topo,
        "--config", str(config_path),
        "--ep", str(episodes),
        "--seeds", str(seeds),
        "--no-train",  # skip RL training during eval-only pass (handled separately)
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=episodes * 3 + 300,  # generous timeout
            cwd=str(ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        dt = time.time() - t0

        if result.returncode != 0:
            return {"label": label, "error": result.stderr[-500:], "time": dt}

        # Parse output — find the results summary lines
        lines = result.stdout.split("\n")
        algorithms = {}
        in_summary = False
        for line in lines:
            if "Results Summary:" in line:
                in_summary = True
                continue
            if in_summary and line.startswith("  ---"):
                continue
            if in_summary and line.strip():
                parts = line.split()
                if len(parts) >= 5 and parts[0] not in ("Algorithm", "Saved:"):
                    try:
                        name = parts[0]
                        cost = float(parts[1])
                        fill = float(parts[2])
                        bw = float(parts[3]) if len(parts) > 3 else 0
                        algorithms[name] = {"cost": cost, "fill": fill, "bw": bw}
                    except (ValueError, IndexError):
                        pass

        return {"label": label, "algorithms": algorithms, "time": dt, "stdout": result.stdout[-2000:]}

    except subprocess.TimeoutExpired:
        return {"label": label, "error": "Timeout", "time": time.time() - t0}
    except Exception as e:
        return {"label": label, "error": str(e), "time": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=200, help="Training episodes")
    parser.add_argument("--seeds", type=int, default=3, help="Eval seeds")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--scenarios", nargs="*", help="Specific scenarios to run (by label)")
    args = parser.parse_args()

    scenarios_to_run = SCENARIOS
    if args.scenarios:
        scenarios_to_run = [s for s in SCENARIOS if s[2] in args.scenarios]

    print(f"Running {len(scenarios_to_run)} scenarios")
    print(f"  Episodes: {args.ep}, Eval seeds: {args.seeds}")
    print(f"  Max parallel workers: {args.max_workers}")
    print(f"  Scenarios: {[s[2] for s in scenarios_to_run]}")
    print()

    # Prepare args
    task_args = [(s[0], s[1], s[2], args.ep, args.seeds) for s in scenarios_to_run]

    results = {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_one_scenario, ta): ta[2] for ta in task_args}
        for future in as_completed(futures):
            label = futures[future]
            try:
                r = future.result()
                results[label] = r
                if "error" in r:
                    print(f"  [{label}] FAILED: {r['error'][:100]}")
                else:
                    algos = r.get("algorithms", {})
                    milp = algos.get("MILP", {})
                    best_rl = max(
                        [(k, v) for k, v in algos.items() if not k.startswith("SEA") and k not in ("MILP", "BaseStock", "(s,S)")],
                        key=lambda x: x[1].get("fill", 0), default=("?", {})
                    )
                    best_sea = max(
                        [(k, v) for k, v in algos.items() if k.startswith("SEA")],
                        key=lambda x: x[1].get("fill", 0), default=("?", {})
                    )
                    print(f"  [{label}] MILP={milp.get('fill',0):.3f} | best_RL={best_rl[0]}={best_rl[1].get('fill',0):.3f} | best_SEA={best_sea[0]}={best_sea[1].get('fill',0):.3f} | t={r.get('time',0):.0f}s")
            except Exception as e:
                results[label] = {"label": label, "error": str(e)}
                print(f"  [{label}] EXCEPTION: {e}")

    total_time = time.time() - t_start

    # Print summary table
    print(f"\n{'='*100}")
    print(f"  ALL SCENARIOS RESULTS  (total time: {total_time:.0f}s)")
    print(f"{'='*100}")
    header = f"{'Scenario':<16} {'MILP':>8} {'BaseStock':>10} {'Best RL':>8} {'RL fill':>8} {'Best SEA':>8} {'SEA fill':>8} {'Δfill':>7}"
    print(header)
    print("-" * 100)
    for s in SCENARIOS:
        label = s[2]
        r = results.get(label, {})
        algos = r.get("algorithms", {})

        milp = algos.get("MILP", {})
        bs = algos.get("BaseStock", {})
        rl_algos = {k: v for k, v in algos.items() if k in ("IQL", "QMIX", "MAPPO")}
        sea_algos = {k: v for k, v in algos.items() if k.startswith("SEA")}
        best_rl = max(rl_algos.items(), key=lambda x: x[1].get("fill", 0)) if rl_algos else ("?", {})
        best_sea = max(sea_algos.items(), key=lambda x: x[1].get("fill", 0)) if sea_algos else ("?", {})

        rl_fill = best_rl[1].get("fill", 0)
        sea_fill = best_sea[1].get("fill", 0)
        delta = sea_fill - rl_fill

        print(f"{label:<16} {milp.get('fill',0):>8.4f} {bs.get('fill',0):>10.4f} {best_rl[0]:>8} {rl_fill:>8.4f} {best_sea[0]:>8} {sea_fill:>8.4f} {delta:>+7.4f}")

    # Save results
    out_path = ROOT / "meirp_project" / "results" / "all_scenarios.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for k, v in results.items():
        if "stdout" in v:
            serializable[k] = {kk: vv for kk, vv in v.items() if kk != "stdout"}
        else:
            serializable[k] = v
    json.dump(serializable, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
