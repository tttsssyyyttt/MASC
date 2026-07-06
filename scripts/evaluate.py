"""Unified evaluation entry point."""

from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from meirp_project.core.config import EnvConfig, from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.registry import make_agent
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.baselines.ss_policy import SSPolicy
from meirp_project.evaluation.runner import EvalRunner
from meirp_project.evaluation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate algorithms")
    parser.add_argument("--env", type=str, default="configs/env_3layers.yaml")
    parser.add_argument("--algo", type=str, default="all", help="Algorithm name or 'all' or 'baselines'")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = from_yaml(args.env) if os.path.exists(args.env) else EnvConfig(
        num_nodes=7, edges=[(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)],
        layers=[0,1,1,2,2,2,2],
        H=[1,2,2,3,3,3,3], B=[5,4,4,3,3,3,3],
        seed=args.seed,
    )

    runner = EvalRunner(cfg)
    results = {}
    env = MEIRPEnv(cfg)

    # Evaluate baselines
    if args.algo in ["all", "baselines"]:
        print("\n=== BaseStock ===")
        bs = BaseStockPolicy(cfg)
        results["BaseStock"] = runner.eval_baseline(bs, n_episodes=args.episodes)

        print("\n=== (s,S) Policy ===")
        ss = SSPolicy(cfg)
        results["(s,S)"] = runner.eval_baseline(ss, n_episodes=args.episodes)

    # Evaluate RL algorithms
    rl_algos = ["iql", "qmix", "mappo"]
    obs_dim = env.observation_space.shape[1]

    for algo_name in rl_algos:
        if args.algo in ["all", algo_name]:
            print(f"\n=== {algo_name.upper()} ===")
            agent = make_agent(algo_name, cfg, obs_dim=obs_dim, device=args.device)

            # Try loading checkpoint
            ckpt_path = os.path.join("checkpoints", f"{algo_name}_best.pt")
            if os.path.exists(ckpt_path):
                agent.load(ckpt_path)
                print(f"  Loaded checkpoint: {ckpt_path}")

            results[algo_name.upper()] = runner.eval_agent(agent, n_episodes=args.episodes)

    # Print comparison
    print(f"\n{'='*60}")
    print(f"{'Algorithm':<15} {'Cost/Step':>10} {'Fill Rate':>10} {'Holding':>10} {'Backlog':>10}")
    print("-" * 60)
    for name, m in results.items():
        print(
            f"{name:<15} "
            f"{m.get('avg_cost_per_step', 0):>10.1f} "
            f"{m.get('avg_fill_rate', 0):>10.3f} "
            f"{m.get('total_holding', 0):>10.1f} "
            f"{m.get('total_backlog', 0):>10.1f}"
        )


if __name__ == "__main__":
    main()
