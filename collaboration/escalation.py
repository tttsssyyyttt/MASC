"""Three-level SEA escalation: RL → BaseStock → MILP.

Layer 1 (RL autonomous):   H_t ≤ k  OR  backlog too small to matter
Layer 2 (BaseStock):        H_t > k  AND backlog > demand_mean    (SPC alarm)
Layer 3 (MILP oracle):      H_t > k*1.5                         (extreme, eval only)

Key design:
  - Per-node max-backlog SPC (catches single-node failures)
  - Minimum backlog threshold prevents over-triggering on noise
  - Training: k=3.0 (standard), BaseStock only for real problems
  - Eval:     k=3.0 + MILP oracle for extreme cases
"""

from __future__ import annotations

import numpy as np

from core.config import EnvConfig
from core.health_monitor import HealthMonitor
from baselines.base_stock import BaseStockPolicy


class ThreeLevelEscalation:
    """Three-level SPC-based escalation controller."""

    def __init__(self, cfg: EnvConfig, k_spc: float = 3.0):
        self.cfg = cfg
        self.health = HealthMonitor(cfg.num_nodes)
        self.k = k_spc
        self.basestock = BaseStockPolicy(cfg)
        self._milp = None

        self.steps = 0; self.alarms = 0; self.expert_steps = 0
        self.bs_steps = 0; self.milp_steps = 0

    @property
    def milp(self):
        if self._milp is None:
            from ..collaboration.milp_advisor import MILPAdvisor
            self._milp = MILPAdvisor(self.cfg, planning_horizon=15)
        return self._milp

    def step(self, obs, rl_act, env, demand_forecast=None):
        """Execute one SEA decision step.

        Only intervenes when:
          1. SPC alarm (H_t > k) AND backlog is materially significant
          2. OR extreme case triggers MILP oracle (eval only)

        Returns: (action, source_label, info_dict)
        """
        self.steps += 1

        fill_rates = np.zeros(env.n)
        for i in range(env.n):
            total_need = env.demand[i] + env.backlog[i]
            fill_rates[i] = max(0.0, 1.0 - env.backlog[i] / max(total_need, 1e-9))

        H = self.health.update(env.backlog, env.demand, fill_rates)
        max_bl = float(env.backlog.max())
        min_alarm_bl = max(2.0, self.cfg.demand_mean * 0.5)  # must exceed half mean demand

        info = {"H": H, "rl_act": rl_act, "layer": 1}

        # Layer 1: RL autonomous
        if abs(H) <= self.k or not self.health.is_warm or max_bl < min_alarm_bl:
            return rl_act, "rl", info

        self.alarms += 1

        # Layer 2: BaseStock algorithm expert (SPC alarm + real backlog)
        try:
            bs_act = self.basestock.get_actions(env)
            self.expert_steps += 1
            self.bs_steps += 1
            info["layer"] = 2
            return bs_act, "basestock", info
        except Exception:
            return rl_act, "rl", info

    def step_eval(self, obs, rl_act, env, demand_forecast=None):
        """Evaluation version with MILP oracle for extreme cases."""
        self.steps += 1

        fill_rates = np.zeros(env.n)
        for i in range(env.n):
            total_need = env.demand[i] + env.backlog[i]
            fill_rates[i] = max(0.0, 1.0 - env.backlog[i] / max(total_need, 1e-9))

        H = self.health.update(env.backlog, env.demand, fill_rates)
        max_bl = float(env.backlog.max())
        min_alarm_bl = max(2.0, self.cfg.demand_mean * 0.5)

        info = {"H": H, "rl_act": rl_act, "layer": 1}

        if abs(H) <= self.k or not self.health.is_warm or max_bl < min_alarm_bl:
            return rl_act, "rl", info

        self.alarms += 1

        # Layer 3 first for extreme cases (only with oracle demand)
        if demand_forecast is not None and abs(H) > self.k * 1.5:
            try:
                milp_act = self.milp.get_action(
                    env.inventory, env.backlog, env.pipeline,
                    demand_forecast=demand_forecast
                )
                self.expert_steps += 1
                self.milp_steps += 1
                info["layer"] = 3
                return milp_act, "milp", info
            except Exception:
                pass

        # Layer 2: BaseStock
        try:
            bs_act = self.basestock.get_actions(env)
            self.expert_steps += 1
            self.bs_steps += 1
            info["layer"] = 2
            return bs_act, "basestock", info
        except Exception:
            return rl_act, "rl", info

    def stats(self):
        t = max(self.steps, 1)
        return {
            "spc": self.alarms / t,
            "expert": self.expert_steps / t,
            "bs": self.bs_steps / t,
            "milp": self.milp_steps / t,
            "rl": (t - self.alarms) / t,
        }
