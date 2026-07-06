"""Shared evaluation utilities for tuning and comparison scripts."""

from __future__ import annotations

from typing import Dict

import numpy as np

from meirp_project.core.env import MEIRPEnv


def evaluate_agent(agent, cfg, n_episodes: int = 5, deterministic: bool = True) -> dict:
    """Evaluate a MADRL agent across multiple episodes.

    Args:
        agent: MADRLAgent instance
        cfg: EnvConfig
        n_episodes: number of evaluation episodes
        deterministic: use greedy policy

    Returns:
        metrics dict with mean/std for key metrics
    """
    rewards = []
    holdings = []
    backlogs = []
    fill_rates = []

    for seed in range(n_episodes):
        env = MEIRPEnv(cfg)
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        ep_holding = 0.0
        ep_backlog = 0.0
        ep_fill_rate_sum = 0.0
        steps = 0
        done = False

        while not done:
            actions = agent.get_actions(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            ep_reward += r.sum()
            ep_holding += info["holding_costs"].sum()
            ep_backlog += info["backlog_costs"].sum()
            ep_fill_rate_sum += info["fill_rates"].mean()
            steps += 1

        rewards.append(ep_reward)
        holdings.append(ep_holding)
        backlogs.append(ep_backlog)
        fill_rates.append(ep_fill_rate_sum / steps)

    return {
        "avg_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "avg_holding": np.mean(holdings),
        "avg_backlog": np.mean(backlogs),
        "avg_fill_rate": np.mean(fill_rates),
        "cost_per_step": (np.mean(holdings) + np.mean(backlogs)) / cfg.episode_len,
    }


def evaluate_baseline(policy, cfg, n_episodes: int = 5) -> dict:
    """Evaluate a baseline policy across multiple episodes.

    Args:
        policy: InventoryPolicy instance
        cfg: EnvConfig
        n_episodes: number of evaluation episodes

    Returns:
        metrics dict
    """
    total_rewards = []
    total_holdings = []
    total_backlogs = []
    fill_rates = []

    for seed in range(n_episodes):
        env = MEIRPEnv(cfg)
        env.reset(seed=seed)
        result = policy.run_episode(env)
        total_rewards.append(result["total_reward"])
        total_holdings.append(result["total_holding"])
        total_backlogs.append(result["total_backlog"])
        fill_rates.append(result["avg_fill_rate"])

    return {
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_holding": np.mean(total_holdings),
        "avg_backlog": np.mean(total_backlogs),
        "avg_fill_rate": np.mean(fill_rates),
        "cost_per_step": (np.mean(total_holdings) + np.mean(total_backlogs)) / (cfg.episode_len * n_episodes),
    }


def print_result(name: str, metrics: dict, indent: int = 2):
    """Print evaluation results for one configuration."""
    prefix = " " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  Reward:      {metrics.get('avg_reward', 0):.1f} +/- {metrics.get('std_reward', 0):.1f}")
    print(f"{prefix}  Holding:     {metrics.get('avg_holding', 0):.1f}")
    print(f"{prefix}  Backlog:     {metrics.get('avg_backlog', 0):.1f}")
    print(f"{prefix}  Fill Rate:   {metrics.get('avg_fill_rate', 0):.3f}")
    print(f"{prefix}  Cost/Step:   {metrics.get('cost_per_step', 0):.1f}")
