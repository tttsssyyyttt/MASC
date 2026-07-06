"""Replay buffer for QMIX (stores episode-level transitions)."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import random


class EpisodeBuffer:
    """Stores full episodes for QMIX training."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)

    def push_episode(self, episode: list):
        """Store a complete episode.

        Args:
            episode: list of (obs, actions, rewards, next_obs, dones) tuples
        """
        self.episodes.append(episode)

    def sample(self, batch_size: int, max_steps: int = 30) -> dict:
        """Sample a batch of episodes, truncated to max_steps.

        Returns:
            batch: dict with arrays of shape (batch, max_steps, ...)
        """
        episodes = random.sample(self.episodes, min(batch_size, len(self.episodes)))

        obs_list, actions_list, rewards_list, next_obs_list, dones_list, states_list, next_states_list = [], [], [], [], [], [], []

        for ep in episodes:
            # Truncate or pad to max_steps
            T = min(len(ep), max_steps)
            obs_ep = np.array([ep[t][0] for t in range(T)])
            actions_ep = [ep[t][1] for t in range(T)]
            rewards_ep = np.array([ep[t][2] for t in range(T)])
            next_obs_ep = np.array([ep[t][3] for t in range(T)])
            dones_ep = np.array([ep[t][4] for t in range(T)])

            # State = flattened obs of all agents
            states_ep = obs_ep.reshape(T, -1)
            next_states_ep = next_obs_ep.reshape(T, -1)

            obs_list.append(obs_ep)
            actions_list.append(actions_ep)
            rewards_list.append(rewards_ep)
            next_obs_list.append(next_obs_ep)
            dones_list.append(dones_ep)
            states_list.append(states_ep)
            next_states_list.append(next_states_ep)

        return {
            "obs": np.array(obs_list),
            "actions": actions_list,
            "rewards": np.array(rewards_list),
            "next_obs": np.array(next_obs_list),
            "dones": np.array(dones_list),
            "states": np.array(states_list),
            "next_states": np.array(next_states_list),
        }

    def __len__(self):
        return len(self.episodes)
