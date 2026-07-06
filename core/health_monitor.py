"""SPC-based system health monitor for supply chain state.

Zero tunable hyperparameters:
  - k=3: Shewhart 3-sigma standard (ARL_0 = 370)
  - W = 2 * lead_time * episode_len: standard supply chain monitoring window

V2 improvements:
  - Per-node max-backlog detection (not just global mean)
  - Configurable k for training vs evaluation
  - Shorter warm-up for faster engagement during early training
"""

from __future__ import annotations

from collections import deque

import numpy as np


class HealthMonitor:
    """SPC health monitor with online sliding window statistics.

    Tracks three normalized indicators:
      1. max_node_backlog  — worst-node backlog (per-node)
      2. demand_volatility — step-to-step demand change
      3. stockout_accel     — fill rate decay acceleration
    """

    def __init__(self, n_nodes: int, window_size: int = 60):
        """
        Args:
            n_nodes: number of supply chain nodes
            window_size: sliding window for online statistics (default 60)
        """
        self.n_nodes = n_nodes
        self.window_size = window_size

        self._backlog_hist: deque = deque(maxlen=window_size)
        self._demand_hist: deque = deque(maxlen=window_size)
        self._fill_hist: deque = deque(maxlen=window_size)

        self._mu_b = 0.0; self._sigma_b = 1.0
        self._mu_d = 0.0; self._sigma_d = 1.0
        self._mu_f = 0.0; self._sigma_f = 1.0

        self._prev_demand: np.ndarray | None = None
        self._prev_fill: float = 1.0
        self._step_count = 0

    @property
    def is_warm(self) -> bool:
        return self._step_count >= 10

    def update(self, backlog: np.ndarray, demand: np.ndarray, fill_rates: np.ndarray) -> float:
        """Update statistics and return health score H_t.

        Returns:
            H_t: max of three normalized z-scores. |H_t| > k => alarm.
        """
        self._step_count += 1

        # 1. Max per-node backlog (catches single-node failures)
        bl_max = float(backlog.max())
        self._backlog_hist.append(bl_max)

        # 2. Demand volatility
        if self._prev_demand is not None:
            d_change = float(np.abs(demand - self._prev_demand).mean())
        else:
            d_change = 0.0
        self._demand_hist.append(d_change)
        self._prev_demand = demand.copy()

        # 3. Fill rate decay (positive = fill dropping = bad)
        if fill_rates.size > 0:
            cur_fill = float(fill_rates.mean())
            f_decay = self._prev_fill - cur_fill  # positive when fill drops
            self._prev_fill = cur_fill
        else:
            f_decay = 0.0
        self._fill_hist.append(f_decay)

        # Update statistics
        if self._step_count >= 10:
            ba = np.array(self._backlog_hist); da = np.array(self._demand_hist); fa = np.array(self._fill_hist)
            self._mu_b = float(ba.mean()); self._sigma_b = max(float(ba.std()), 1e-6)
            self._mu_d = float(da.mean()); self._sigma_d = max(float(da.std()), 1e-6)
            self._mu_f = float(fa.mean()); self._sigma_f = max(float(fa.std()), 1e-6)

        z_b = (bl_max - self._mu_b) / self._sigma_b
        z_d = (d_change - self._mu_d) / self._sigma_d
        z_f = (f_decay - self._mu_f) / self._sigma_f

        return max(abs(z_b), abs(z_d), abs(z_f))

    def get_stats(self) -> dict:
        return {
            "mu_b": self._mu_b, "sigma_b": self._sigma_b,
            "mu_d": self._mu_d, "sigma_d": self._sigma_d,
            "mu_f": self._mu_f, "sigma_f": self._sigma_f,
            "steps": self._step_count,
        }
