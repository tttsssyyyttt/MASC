"""Unified training entry point for all MADRL algorithms."""

from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from meirp_project.core.config import EnvConfig, from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.registry import make_agent


def main():
    parser = argparse.ArgumentParser(description="Train MADRL agent")
    parser.add_argument("--env", type=str, default="configs/env_3layers.yaml", help="Env config YAML")
    parser.add_argument("--algo", type=str, default="iql", choices=["iql", "qmix", "mappo"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Load config
    cfg = from_yaml(args.env) if os.path.exists(args.env) else EnvConfig(
        num_nodes=7, edges=[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        layers=[0, 1, 1, 2, 2, 2, 2],
        H=[1, 2, 2, 3, 3, 3, 3], B=[5, 4, 4, 3, 3, 3, 3],
        seed=args.seed,
    )
    cfg.seed = args.seed

    # Create environment
    env = MEIRPEnv(cfg)
    obs_dim = env.observation_space.shape[1]

    # Create agent
    agent = make_agent(
        args.algo, cfg, obs_dim=obs_dim,
        hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma,
        device=args.device,
    )

    # Detect on-policy vs off-policy
    is_on_policy = hasattr(agent, 'buffer') and hasattr(agent.buffer, 'compute_returns')

    print(f"Training {args.algo.upper()} for {args.episodes} episodes on {cfg.num_nodes}-node network")

    episode_rewards = []
    best_reward = -float("inf")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            actions = agent.get_actions(obs)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # Store transition
            if hasattr(agent, 'store_transition'):
                dones_arr = np.full(cfg.num_nodes, float(terminated))
                agent.store_transition(obs, actions, rewards, next_obs, dones_arr)

            # Off-policy (IQL): update per-step after warmup
            if not is_on_policy and hasattr(agent, 'update'):
                if getattr(agent, 'total_steps', 0) >= 1000:
                    agent.update()

            ep_reward += rewards.sum()
            obs = next_obs

        # On-policy (MAPPO): update at end of episode
        if is_on_policy and hasattr(agent, 'update'):
            if len(getattr(agent.buffer, 'rewards', [])) > 0:
                agent.update()

        episode_rewards.append(ep_reward)

        # Log
        if (ep + 1) % 50 == 0:
            avg = sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
            print(f"Episode {ep + 1}/{args.episodes} | Avg Reward (last 50): {avg:.1f}")

        # Save best
        if ep_reward > best_reward:
            best_reward = ep_reward
            os.makedirs(args.save_dir, exist_ok=True)
            agent.save(os.path.join(args.save_dir, f"{args.algo}_best.pt"))

    # Final save
    os.makedirs(args.save_dir, exist_ok=True)
    agent.save(os.path.join(args.save_dir, f"{args.algo}_final.pt"))
    print(f"Training complete. Best reward: {best_reward:.1f}")


if __name__ == "__main__":
    main()
