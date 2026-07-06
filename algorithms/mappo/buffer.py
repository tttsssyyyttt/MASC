"""Rollout buffer for MAPPO (on-policy) with BDQ actions.

Stores per-branch order quantities instead of (q, alpha).
Actions format:
  - Root nodes (no upstream): int q → stored as branch 0
  - Non-root nodes: list [q_up0, q_up1, ...] → stored per branch
"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """Stores a single rollout for on-policy training with BDQ actions."""

    def __init__(self):
        self.obs = []
        self.actions = []  # raw action list per step
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def push(
            self,
            obs: np.ndarray,
            actions: list,
            rewards: np.ndarray,
            values: np.ndarray,
            log_probs: np.ndarray,
            dones: np.ndarray,
    ):
        self.obs.append(obs.copy())
        self.actions.append(actions)
        self.rewards.append(rewards.copy())
        self.values.append(values.copy())
        self.log_probs.append(log_probs.copy())
        self.dones.append(dones.copy())

    def get_actions_flat(self):
        """Flatten actions into per-branch arrays for GNN MAPPO training.

        Returns:
            dict mapping (agent_idx, branch_idx) → (T*N,) int array
        """
        T = len(self.actions)
        result = {}

        if T == 0:
            return result

        def n_branches_for(action):
            if isinstance(action, dict):
                return 1 + len(action.get("alpha", []))
            if isinstance(action, (list, np.ndarray)):
                return len(action) + 1
            return 1

        def branch_value(action, branch_idx):
            if isinstance(action, dict):
                if branch_idx == 0:
                    return int(action.get("q", 0))
                alpha = action.get("alpha", [])
                return int(alpha[branch_idx - 1]) if branch_idx - 1 < len(alpha) else 0
            if isinstance(action, (list, np.ndarray)):
                if branch_idx == 0:
                    return int(sum(action))
                return int(action[branch_idx - 1]) if branch_idx - 1 < len(action) else 0
            return int(action) if branch_idx == 0 else 0

        first_actions = self.actions[0]
        for i, a in enumerate(first_actions):
            n_branches = n_branches_for(a)

            for j in range(n_branches):
                key = (i, j)
                arr = np.zeros(T * len(first_actions), dtype=np.int64)
                idx = 0
                for t in range(T):
                    for agent_i, a in enumerate(self.actions[t]):
                        if agent_i == i:
                            arr[idx] = branch_value(a, j)
                        idx += 1
                result[key] = arr

        return result

    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95):
        """Compute GAE advantages and returns."""
        T = len(self.rewards)
        N = self.rewards[0].shape[0]

        self.advantages = np.zeros((T, N), dtype=np.float64)
        self.returns = np.zeros((T, N), dtype=np.float64)

        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = np.zeros(N)
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = last_gae + self.values[t]

    def get_batches(self, batch_size: int, n_agents: int, obs_dim: int):
        """Yield mini-batches for training.

        Yields:
            dict with full graph obs, agent indices, actions, old log_probs, returns, advantages
        """
        T = len(self.obs)

        log_probs_flat = np.concatenate(self.log_probs, axis=0)
        returns_flat = self.returns.reshape(-1)
        adv_flat = self.advantages.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Flatten actions per branch
        actions_flat = self.get_actions_flat()

        total = T * n_agents
        indices = np.arange(total)
        np.random.shuffle(indices)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]
            time_idx = idx // n_agents
            agent_idx = idx % n_agents
            obs_context = np.stack([self.obs[t] for t in time_idx], axis=0)

            batch = {
                "obs": torch.FloatTensor(obs_context),
                "agent_indices": torch.LongTensor(agent_idx),
                "old_log_probs": torch.FloatTensor(log_probs_flat[idx]),
                "returns": torch.FloatTensor(returns_flat[idx]),
                "advantages": torch.FloatTensor(adv_flat[idx]),
                "actions_q_per_branch": {},
            }

            for key, arr in actions_flat.items():
                batch["actions_q_per_branch"][key] = torch.LongTensor(arr[idx])

            yield batch

    def __len__(self):
        return len(self.obs)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
