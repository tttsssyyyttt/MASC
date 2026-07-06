"""Unified evaluation runner for all algorithms and baselines."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.config import EnvConfig
from ..core.env import MEIRPEnv
from ..algorithms.base import MADRLAgent
from .metrics import compute_metrics


class EvalRunner:
    """Run evaluation across multiple seeds and collect metrics."""

    def __init__(self, cfg: EnvConfig, n_seeds: int = 5):
        self.cfg = cfg
        self.n_seeds = n_seeds

    def eval_agent(
        self,
        agent: MADRLAgent,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> dict:
        """Evaluate a MADRL agent.

        Returns:
            results: dict with metrics averaged over episodes
        """
        all_metrics = []

        for ep in range(n_episodes):
            env = MEIRPEnv(self.cfg)
            obs, _ = env.reset(seed=ep)
            info_history = []
            done = False

            while not done:
                actions = agent.get_actions(obs, deterministic=deterministic)
                obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                info_history.append(info)

            metrics = compute_metrics(info_history, self.cfg)
            all_metrics.append(metrics)

        return self._aggregate(all_metrics)

    def eval_baseline(self, baseline, n_episodes: int = 10) -> dict:
        """Evaluate a baseline policy (BaseStock, sS, etc.).

        The baseline must have a `run_episode(env)` method.
        """
        all_metrics = []

        for ep in range(n_episodes):
            env = MEIRPEnv(self.cfg)
            obs, _ = env.reset(seed=ep)
            info_history = []
            done = False

            while not done:
                actions = baseline.get_actions(env)
                obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                info_history.append(info)

            metrics = compute_metrics(info_history, self.cfg)
            all_metrics.append(metrics)

        return self._aggregate(all_metrics)

    def _aggregate(self, all_metrics: list) -> dict:
        """Aggregate metrics across episodes."""
        if not all_metrics:
            return {}

        keys = all_metrics[0].keys()
        result = {}

        for key in keys:
            values = [m[key] for m in all_metrics if key in m]
            if len(values) == 0:
                continue

            if isinstance(values[0], (int, float)):
                result[key] = np.mean(values)
                result[f"{key}_std"] = np.std(values)
            elif isinstance(values[0], np.ndarray):
                result[key] = np.mean(values, axis=0)
            else:
                result[key] = values

        result["n_episodes"] = len(all_metrics)
        return result
