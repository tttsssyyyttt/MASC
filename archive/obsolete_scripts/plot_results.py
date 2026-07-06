"""Batch generate paper figures from evaluation results.

Produces: training curves, cost breakdowns, comparison tables,
attention heatmaps, network topology diagrams, bullwhip effect charts.
"""

from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import yaml

from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.registry import make_agent
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.baselines.ss_policy import SSPolicy
from meirp_project.evaluation.visualization import (
    plot_training_curve,
    plot_cost_breakdown,
    plot_comparison_table,
    plot_attention_heatmap,
    plot_network_topology,
)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--env", type=str, default="configs/env_3layers.yaml")
    parser.add_argument("--experiment", type=str, default="configs/experiment.yaml")
    parser.add_argument("--output", type=str, default="paper_figures")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load config
    if os.path.exists(args.env):
        cfg = from_yaml(args.env)
    else:
        cfg = EnvConfig(
            num_nodes=7, edges=[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
            layers=[0, 1, 1, 2, 2, 2, 2],
            H=[1, 2, 2, 3, 3, 3, 3], B=[5, 4, 4, 3, 3, 3, 3],
        )

    obs_dim = 3 + 3 * cfg.lead_time
    runner = EvalRunner(cfg, n_seeds=1)
    results = {}

    # Evaluate baselines
    print("Evaluating BaseStock...")
    bs = BaseStockPolicy(cfg)
    results["BaseStock"] = runner.eval_baseline(bs, n_episodes=args.episodes)

    print("Evaluating (s,S)...")
    ss = SSPolicy(cfg)
    results["(s,S)"] = runner.eval_baseline(ss, n_episodes=args.episodes)

    # Evaluate RL algorithms (with and without GNN)
    for algo_name in ["iql", "qmix", "mappo"]:
        for use_gnn in [False, True]:
            suffix = "+GNN" if use_gnn else ""
            key = f"{algo_name.upper()}{suffix}"
            print(f"Evaluating {key}...")
            agent = make_agent(algo_name, cfg, obs_dim=obs_dim, use_gnn=use_gnn)
            results[key] = runner.eval_agent(agent, n_episodes=args.episodes)

            # Attention heatmap for GNN models
            if use_gnn and agent.encoder is not None:
                print(f"  Generating attention heatmap for {key}...")
                import torch
                env = MEIRPEnv(cfg)
                obs, _ = env.reset()
                # Get attention weights from encoder
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    _ = agent.encoder(obs_t)
                    # Get from GAT layers
                    gat_encoder = agent.encoder.gat
                    attn = gat_encoder.get_attention_weights()
                    if attn is not None:
                        attn_np = attn.cpu().numpy()
                        edge_index = None
                        # Build edge_index from cfg
                        src = [e[0] for e in cfg.edges]
                        dst = [e[1] for e in cfg.edges]
                        all_src = src + dst
                        all_dst = dst + src
                        edge_index = np.array([all_src, all_dst])

                        plot_attention_heatmap(
                            attn_np, edge_index,
                            title=f"Attention Weights - {key}",
                            save_path=os.path.join(args.output, f"attention_{key}.png"),
                        )

    # Generate comparison table
    print("Generating comparison table...")
    plot_comparison_table(
        results,
        save_path=os.path.join(args.output, "comparison_table.png"),
    )

    # Cost breakdown for each algorithm
    for name, metrics in results.items():
        plot_cost_breakdown(
            metrics,
            title=f"Cost Breakdown - {name}",
            save_path=os.path.join(args.output, f"cost_{name}.png"),
        )

    # Network topology
    print("Generating network topology...")
    plot_network_topology(
        edges=cfg.edges,
        layers=cfg.layers,
        title="Supply Chain Network Topology",
        save_path=os.path.join(args.output, "network_topology.png"),
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'Algorithm':<15} {'Cost/Step':>10} {'Fill Rate':>10} {'Bullwhip':>10} {'Turnover':>10}")
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<15} "
            f"{m.get('avg_cost_per_step', 0):>10.1f} "
            f"{m.get('avg_fill_rate', 0):>10.3f} "
            f"{m.get('avg_bullwhip_cv', 0):>10.3f} "
            f"{m.get('avg_inventory_turnover', 0):>10.2f}"
        )

    print(f"\nFigures saved to: {args.output}")


if __name__ == "__main__":
    main()
