"""QMIX training loop."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .agent import QMIXAgent
from ...core.env import MEIRPEnv


class QMIXTrainer:
    """Handles the training loop for QMIX."""

    def __init__(
        self,
        agent: QMIXAgent,
        env: MEIRPEnv,
        max_episodes: int = 1000,
        eval_interval: int = 50,
        eval_episodes: int = 5,
        warmup_episodes: int = 10,
        log_interval: int = 10,
    ):
        self.agent = agent
        self.env = env
        self.max_episodes = max_episodes
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.warmup_episodes = warmup_episodes
        self.log_interval = log_interval

    def train(self) -> dict:
        """Run training loop.

        Returns:
            history: dict with 'episode_rewards', 'eval_rewards' lists
        """
        episode_rewards = []
        eval_rewards = []
        best_eval_reward = -float("inf")

        for ep in range(self.max_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                actions = self.agent.get_actions(obs)

                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated

                dones = np.full(self.env.n, float(terminated))
                self.agent.store_transition(obs, actions, rewards, next_obs, dones)

                ep_reward += rewards.sum()
                obs = next_obs

            episode_rewards.append(ep_reward)

            # Update after each episode (QMIX uses episodic replay)
            if ep >= self.warmup_episodes:
                metrics = self.agent.update()

            # Log
            if (ep + 1) % self.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-self.log_interval:])
                epsilon = self.agent._get_epsilon()
                print(f"Episode {ep+1}/{self.max_episodes} | "
                      f"Avg Reward: {avg_reward:.1f} | Epsilon: {epsilon:.3f}")

            # Evaluate
            if (ep + 1) % self.eval_interval == 0:
                eval_reward = self._evaluate()
                eval_rewards.append(eval_reward)

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.agent.save("checkpoints/qmix_best.pt")

                print(f"  Eval Reward: {eval_reward:.1f} | Best: {best_eval_reward:.1f}")

        return {
            "episode_rewards": episode_rewards,
            "eval_rewards": eval_rewards,
        }

    def _evaluate(self) -> float:
        """Run evaluation episodes with deterministic policy."""
        total_rewards = []
        for _ in range(self.eval_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                actions = self.agent.get_actions(obs, deterministic=True)
                obs, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                ep_reward += rewards.sum()

            total_rewards.append(ep_reward)

        return np.mean(total_rewards)
