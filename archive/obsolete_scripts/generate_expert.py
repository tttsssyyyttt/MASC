"""Generate MILP expert data for SEA advisor training.

Runs MILP solver on multiple scenarios and saves (state, expert_action) pairs.
"""

from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from meirp_project.core.config import EnvConfig, from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.solvers.milp_solver import MILPSolver


def _to_bdq_format(actions, upstream, max_order):
    """Convert solver (q, alpha) action format to BDQ format for env.step().

    Args:
        actions: list of (q, alpha) tuples per agent
        upstream: list of upstream node lists per agent
        max_order: max order per node

    Returns:
        bdq_actions: root → int, non-root → list[int]
    """
    bdq = []
    for i, (q, alpha) in enumerate(actions):
        k = len(upstream[i])
        if k == 0:
            bdq.append(min(int(q), max_order))
        else:
            q_per_up = []
            remaining = min(int(q), max_order)
            for j in range(k - 1):
                a = int(round(alpha[j] * remaining))
                q_per_up.append(min(a, remaining))
                remaining -= a
            q_per_up.append(min(remaining, max_order))
            bdq.append(q_per_up)
    return bdq


def main():
    parser = argparse.ArgumentParser(description="Generate MILP expert data")
    parser.add_argument("--env", type=str, default="configs/env_3layers.yaml")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output", type=str, default="expert_data")
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.env):
        cfg = from_yaml(args.env)
    else:
        cfg = EnvConfig(
            num_nodes=7, edges=[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
            layers=[0, 1, 1, 2, 2, 2, 2],
            H=[1, 2, 2, 3, 3, 3, 3], B=[5, 4, 4, 3, 3, 3, 3],
        )

    from meirp_project.core.network import build_network
    upstream, _, _, _ = build_network(cfg)

    solver = MILPSolver(cfg, planning_horizon=args.horizon)
    os.makedirs(args.output, exist_ok=True)

    all_obs = []
    all_bdq_actions = []
    all_inventory = []
    all_backlog = []

    for ep in range(args.n_episodes):
        env = MEIRPEnv(cfg)
        obs, _ = env.reset(seed=ep)

        for step in range(cfg.episode_len):
            # Generate demand forecast for MILP
            demand_forecast = np.random.poisson(
                cfg.demand_mean,
                size=(args.horizon, env.n_terminals)
            ).astype(float)

            # Get expert action from MILP (returns old-format (q, alpha) tuples)
            expert_actions = solver.get_actions(
                env.inventory, env.backlog, env.pipeline, demand_forecast
            )

            # Convert to BDQ format for env.step()
            bdq_actions = _to_bdq_format(expert_actions, upstream, cfg.max_order)

            # Record
            all_obs.append(obs.copy())
            all_bdq_actions.append(bdq_actions)
            all_inventory.append(env.inventory.copy())
            all_backlog.append(env.backlog.copy())

            # Step with BDQ actions
            obs, rewards, terminated, truncated, info = env.step(bdq_actions)
            if terminated:
                break

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{args.n_episodes}: {len(all_obs)} total transitions")

    # Save all data
    save_path = os.path.join(args.output, "expert_data.npz")
    np.savez_compressed(
        save_path,
        obs=np.stack(all_obs, axis=0),
        inventory=np.stack(all_inventory, axis=0),
        backlog=np.stack(all_backlog, axis=0),
        n_nodes=cfg.num_nodes,
        obs_dim=all_obs[0].shape[1],
        allow_pickle=True,
    )
    # Save BDQ actions separately (variable-length nested lists)
    import pickle
    actions_path = os.path.join(args.output, "expert_actions.pkl")
    with open(actions_path, "wb") as f:
        pickle.dump(all_bdq_actions, f)

    print(f"Saved {len(all_obs)} expert transitions to {save_path}")
    print(f"Saved BDQ actions to {actions_path}")


if __name__ == "__main__":
    main()
