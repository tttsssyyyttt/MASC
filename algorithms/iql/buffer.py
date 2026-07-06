"""Replay buffer for IQL (and other value-based methods).

Stores BDQ actions: per-agent list of per-upstream order quantities.
- Root nodes: int q
- Non-root nodes: list [q_up0, q_up1, ...]
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import random


class ReplayBuffer:
    """Simple replay buffer storing (obs, actions, rewards, next_obs, dones) transitions."""

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,          # (N, obs_dim)
        actions: list,            # list of per-agent actions (int or list[int])
        rewards: np.ndarray,      # (N,)
        next_obs: np.ndarray,     # (N, obs_dim)
        dones: np.ndarray,        # (N,)
    ):
        self.buffer.append({
            "obs": obs.copy(),
            "actions": actions,
            "rewards": rewards.copy(),
            "next_obs": next_obs.copy(),
            "dones": dones.copy(),
        })

    def sample(self, batch_size: int) -> dict:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return {
            "obs": np.array([t["obs"] for t in batch]),
            "actions": [t["actions"] for t in batch],
            "rewards": np.array([t["rewards"] for t in batch]),
            "next_obs": np.array([t["next_obs"] for t in batch]),
            "dones": np.array([t["dones"] for t in batch]),
        }

    def __len__(self):
        return len(self.buffer)
