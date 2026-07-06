"""Demand generation for terminal nodes."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator

from .config import EnvConfig


class DemandGenerator:
    """Generates demand for each terminal node per step.

    Supports: poisson, normal, merton, seasonal, trend, shock.
    Non-stationary patterns (seasonal/trend/shock) use an internal
    step counter that advances on each generate() call.
    """

    def __init__(self, cfg: EnvConfig, rng: Optional[Generator] = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self._step = 0

    def reset_step(self):
        """Reset internal step counter (for non-stationary patterns)."""
        self._step = 0

    def generate(self, n_terminals: int) -> np.ndarray:
        """Generate demand for n_terminals in one step."""
        cfg = self.cfg
        t = self._step
        self._step += 1

        if cfg.demand_type == "poisson":
            demands = self.rng.poisson(lam=cfg.demand_mean, size=n_terminals)
        elif cfg.demand_type == "normal":
            demands = self.rng.normal(loc=cfg.demand_mean, scale=cfg.demand_std, size=n_terminals)
            demands = np.maximum(demands, 0)
        elif cfg.demand_type == "merton":
            base = self.rng.normal(loc=cfg.demand_mean, scale=cfg.demand_std * 0.5, size=n_terminals)
            jump_mask = self.rng.random(n_terminals) < 0.1
            jump_size = self.rng.exponential(scale=cfg.demand_mean, size=n_terminals)
            demands = base + jump_mask * jump_size
            demands = np.maximum(demands, 0)
        elif cfg.demand_type == "seasonal":
            lam = max(1, cfg.demand_mean / 2 + cfg.demand_mean / 2 * np.sin(2 * np.pi * t / 15))
            demands = self.rng.poisson(lam=lam, size=n_terminals)
        elif cfg.demand_type == "trend":
            start = max(1, cfg.demand_mean / 3)
            end = cfg.demand_mean * 5 / 3
            lam = start + (end - start) * t / max(cfg.episode_len - 1, 1)
            demands = self.rng.poisson(lam=lam, size=n_terminals)
        elif cfg.demand_type == "shock":
            lam = cfg.demand_mean * 3.0 if t == 15 else cfg.demand_mean
            demands = self.rng.poisson(lam=lam, size=n_terminals)
        else:
            raise ValueError(f"Unknown demand_type: {cfg.demand_type}")

        return demands.astype(np.float64)



