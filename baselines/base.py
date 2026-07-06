"""Shared base class for inventory policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.config import EnvConfig
from ..core.network import build_network
from ..core.env import MEIRPEnv


class InventoryPolicy(ABC):
    """Abstract base class for classical inventory policies."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.upstream, self.downstream, self.topo_order, self.terminals = build_network(cfg)

    @abstractmethod
    def get_actions(self, env: MEIRPEnv) -> list:
        """Compute actions given current env state.

        Returns:
            actions: list of per-agent actions in BDQ format
        """
        pass

    def run_episode(self, env: MEIRPEnv) -> dict:
        """Run a full episode and return metrics."""
        obs, _ = env.reset()
        total_reward = 0.0
        total_holding = 0.0
        total_backlog = 0.0
        total_transport = 0.0
        total_fill_rate = 0.0
        steps = 0

        while True:
            actions = self.get_actions(env)
            obs, rewards, terminated, truncated, info = env.step(actions)
            total_reward += rewards.sum()
            total_holding += info["holding_costs"].sum()
            total_backlog += info["backlog_costs"].sum()
            total_transport += info.get("total_transport_cost", 0.0)
            total_fill_rate += info["fill_rates"].mean()
            steps += 1
            if terminated:
                break

        return {
            "total_reward": total_reward,
            "avg_reward": total_reward / steps,
            "total_holding": total_holding,
            "total_backlog": total_backlog,
            "total_transport": total_transport,
            "avg_fill_rate": total_fill_rate / steps,
        }
