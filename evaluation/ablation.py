"""Ablation experiment runner — MADRL / +GNN / +SEA / +GNN+SEA."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.config import EnvConfig
from ..core.env import MEIRPEnv
from ..algorithms.registry import make_agent
from ..collaboration.sea_wrapper import SEAWrapper
from .runner import EvalRunner
from .metrics import compute_metrics


class AblationRunner:
    """Run ablation experiments across different configurations."""

    def __init__(self, cfg: EnvConfig, algorithm: str = "iql", n_episodes: int = 10):
        self.cfg = cfg
        self.algorithm = algorithm
        self.n_episodes = n_episodes

    def run(self) -> Dict[str, dict]:
        """Run all 4 ablation groups.

        Returns:
            results: dict with keys "MADRL", "+GNN", "+SEA", "+GNN+SEA"
        """
        groups = [
            ("MADRL", False, False),
            ("+GNN", True, False),
            ("+SEA", False, True),
            ("+GNN+SEA", True, True),
        ]

        results = {}
        runner = EvalRunner(self.cfg)
        obs_dim = 3 + 3 * self.cfg.lead_time

        for name, use_gnn, use_sea in groups:
            print(f"\nRunning ablation: {name} (use_gnn={use_gnn}, use_sea={use_sea})")

            # Create agent with correct GNN flag
            agent = make_agent(self.algorithm, self.cfg, obs_dim=obs_dim, use_gnn=use_gnn)

            # Wrap with SEA if needed
            if use_sea:
                from ..collaboration.advisor import ExpertAdvisor
                sea = ExpertAdvisor(self.cfg, intervention_threshold=0.2)
                agent = SEAWrapper(agent, sea=sea, use_sea=True)

            # Evaluate
            metrics = runner.eval_agent(agent, n_episodes=self.n_episodes)
            results[name] = metrics

            print(f"  Total cost: {metrics.get('avg_cost_per_step', 'N/A'):.1f}")
            print(f"  Fill rate: {metrics.get('avg_fill_rate', 'N/A'):.3f}")

        return results

    def compare(self, results: Dict[str, dict]) -> str:
        """Generate comparison table.

        Returns:
            table: formatted string table
        """
        header = f"{'Group':<12} {'Cost/Step':>10} {'Fill Rate':>10} {'Holding':>10} {'Backlog':>10}"
        lines = [header, "-" * len(header)]

        for name in ["MADRL", "+GNN", "+SEA", "+GNN+SEA"]:
            if name in results:
                m = results[name]
                line = (
                    f"{name:<12} "
                    f"{m.get('avg_cost_per_step', 0):>10.1f} "
                    f"{m.get('avg_fill_rate', 0):>10.3f} "
                    f"{m.get('total_holding', 0):>10.1f} "
                    f"{m.get('total_backlog', 0):>10.1f}"
                )
                lines.append(line)

        return "\n".join(lines)
