"""Run ablation experiments: MADRL / +GNN / +SEA / +GNN+SEA."""

from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from meirp_project.core.config import EnvConfig, from_yaml
from meirp_project.evaluation.ablation import AblationRunner


def main():
    parser = argparse.ArgumentParser(description="Ablation experiments")
    parser.add_argument("--env", type=str, default="configs/env_3layers.yaml")
    parser.add_argument("--algo", type=str, default="iql", choices=["iql", "qmix", "mappo"])
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    cfg = from_yaml(args.env) if os.path.exists(args.env) else EnvConfig(
        num_nodes=7, edges=[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        layers=[0, 1, 1, 2, 2, 2, 2],
        H=[1, 2, 2, 3, 3, 3, 3], B=[5, 4, 4, 3, 3, 3, 3],
    )

    runner = AblationRunner(cfg, algorithm=args.algo, n_episodes=args.episodes)
    results = runner.run()

    # Print comparison table
    table = runner.compare(results)
    print(f"\n{table}")


if __name__ == "__main__":
    main()
