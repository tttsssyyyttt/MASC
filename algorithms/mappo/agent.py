"""MAPPO Agent with BDQ branching and optional GNN encoder.

BDQ Actor design:
- Shared backbone + per-upstream branch heads
- Each branch outputs logits for Categorical(max_order+1)
- Action = sample per branch → (q_up0, q_up1, ...)
- log_prob = sum of per-branch log_probs
- entropy = sum of per-branch entropies
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..base import MADRLAgent
from .buffer import RolloutBuffer
from ...core.config import EnvConfig
from ...core.network import build_network
from ...models.encoders import make_encoder


class BranchingActor(nn.Module):
    """BDQ Actor: shared backbone + per-upstream Categorical heads."""

    def __init__(self, input_dim: int, n_branches: int, max_order: int, hidden_dim: int = 128):
        super().__init__()
        self.n_branches = max(n_branches, 1)  # root nodes get 1 branch

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.branches = nn.ModuleList([
            nn.Linear(hidden_dim, max_order + 1)
            for _ in range(self.n_branches)
        ])

    def forward(self, obs: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of (batch, max_order+1) logit tensors per branch."""
        features = self.backbone(obs)
        return [branch(features) for branch in self.branches]


class Critic(nn.Module):
    """Critic taking per-agent features."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class MAPPOAgent(MADRLAgent):
    """MAPPO agent with BDQ branching, centralized critic, and optional GNN encoder."""

    def __init__(
        self,
        cfg: EnvConfig,
        obs_dim: int,
        hidden_dim: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs_per_update: int = 3,
        mini_batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        use_gnn: bool = False,
        edge_index: Optional[torch.Tensor] = None,
        layer_ids: Optional[torch.Tensor] = None,
        n_supply_layers: Optional[int] = None,
    ):
        self.cfg = cfg
        self.n = cfg.num_nodes
        self.max_order = cfg.max_order
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.use_gnn = use_gnn
        self.obs_dim = obs_dim

        self.upstream, self.downstream, self.topo_order, self.terminals = build_network(cfg)
        self.n_upstream = [len(self.upstream[i]) for i in range(self.n)]
        self.n_action_branches = [1 if k == 0 else k + 1 for k in self.n_upstream]

        # Encoder: always use GRU→MLP (same as IQL)
        self.encoder = make_encoder(
            obs_dim, hidden_dim, use_gnn=use_gnn,
            edge_index=edge_index, layer_ids=layer_ids,
            n_supply_layers=n_supply_layers or len(set(cfg.layers)),
            lead_time=cfg.lead_time,
        ).to(self.device)
        self.feature_dim = hidden_dim

        # Branching actors
        self.actors = nn.ModuleList([
            BranchingActor(self.feature_dim, self.n_action_branches[i], cfg.max_order, hidden_dim).to(self.device)
            for i in range(self.n)
        ])

        # Critic
        self.critic = Critic(self.feature_dim, hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(
            list(self.actors.parameters()) + list(self.encoder.parameters()),
            lr=lr_actor
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.total_steps = 0
        self.buffer = RolloutBuffer()
        self._last_encoded_obs = None

    def _encode_obs(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            encoded = self.encoder(obs_t)
        return encoded.squeeze(0).cpu().numpy()

    def _encode_obs_tensor(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_t)

    def get_actions(self, obs: np.ndarray, deterministic: bool = False) -> list:
        """Get BDQ actions: per-upstream order quantities."""
        actions = []
        log_probs = np.zeros(self.n)
        values = np.zeros(self.n)

        encoded = self._encode_obs(obs)
        self._last_encoded_obs = encoded
        self._last_raw_obs = obs.copy()

        # Compute values
        for i in range(self.n):
            enc_t = torch.FloatTensor(encoded[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                v = self.critic(enc_t).squeeze().cpu().numpy()
                values[i] = v

        # Compute actions and log_probs
        for i in range(self.n):
            enc_t = torch.FloatTensor(encoded[i]).unsqueeze(0).to(self.device)
            k = self.n_upstream[i]

            with torch.no_grad():
                branch_logits = self.actors[i](enc_t)

            action_vals = []
            node_log_prob = 0.0

            for j, logits in enumerate(branch_logits):
                probs = torch.softmax(logits, dim=-1)

                if deterministic:
                    q = probs.argmax(dim=-1).item()
                    lp = torch.log(probs[0, q] + 1e-8).item()
                else:
                    dist = torch.distributions.Categorical(probs)
                    q_tensor = dist.sample()
                    q = q_tensor.item()
                    lp = dist.log_prob(q_tensor).item()

                action_vals.append(q)
                node_log_prob += lp

            if k == 0:
                action_i = action_vals[0]  # single int for root nodes
            else:
                action_i = {"q": action_vals[0], "alpha": action_vals[1:]}

            actions.append(action_i)
            log_probs[i] = node_log_prob

        self._last_values = values
        self._last_log_probs = log_probs

        return actions

    def get_entropy(self, obs: np.ndarray) -> np.ndarray:
        """Compute per-node entropy from policy distributions."""
        encoded = self._encode_obs(obs)
        entropies = np.zeros(self.n)

        with torch.no_grad():
            for i in range(self.n):
                enc_t = torch.FloatTensor(encoded[i]).unsqueeze(0).to(self.device)
                branch_logits = self.actors[i](enc_t)
                node_entropy = 0.0
                for logits in branch_logits:
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    node_entropy += dist.entropy().sum().item()
                entropies[i] = node_entropy

        return entropies

    def store_transition(self, obs, actions, rewards, next_obs, dones):
        self.buffer.push(
            obs=obs,
            actions=actions,
            rewards=rewards,
            values=self._last_values,
            log_probs=self._last_log_probs,
            dones=dones,
        )
        self.total_steps += 1

    def _compute_new_log_probs_and_entropy(self, obs_b, actions_q_per_branch, agent_indices_t):
        """Compute new log_probs and entropy for a mini-batch.

        Args:
            obs_b: (B, feature_dim) observations
            actions_q_per_branch: dict mapping (agent_idx, branch_idx) → (B,) LongTensor
            agent_indices_t: (B,) LongTensor agent indices

        Returns:
            new_log_probs: (B,) tensor
            entropy_sum: float
        """
        B = obs_b.size(0)
        new_log_probs = torch.zeros(B, device=self.device)
        entropy_sum = 0.0

        for i in range(self.n):
            mask = (agent_indices_t == i)
            if mask.sum() == 0:
                continue
            obs_i = obs_b[mask]

            branch_logits = self.actors[i](obs_i)

            for j, logits in enumerate(branch_logits):
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                key = (i, j)
                q_j = actions_q_per_branch[key][mask]
                new_log_probs[mask] += dist.log_prob(q_j)
                entropy_sum += dist.entropy().mean().item()

        return new_log_probs, entropy_sum

    def update(self, batch: Optional[dict] = None) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {"loss": 0.0}

        self.buffer.compute_returns(self.gamma, self.gae_lambda)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.epochs_per_update):
            for mini_batch in self.buffer.get_batches(
                self.mini_batch_size, self.n, self.feature_dim,
            ):
                obs_context_b = mini_batch["obs"].to(self.device)
                agent_indices = mini_batch["agent_indices"].to(self.device)
                old_log_probs_b = mini_batch["old_log_probs"].to(self.device)
                returns_b = mini_batch["returns"].to(self.device)
                advantages_b = mini_batch["advantages"].to(self.device)
                actions_q_per_branch = mini_batch["actions_q_per_branch"]  # dict

                B = obs_context_b.size(0)
                encoded_all = self._encode_obs_tensor(obs_context_b)
                obs_b = encoded_all[torch.arange(B, device=self.device), agent_indices]

                # Move branch actions to device
                for key in actions_q_per_branch:
                    actions_q_per_branch[key] = actions_q_per_branch[key].to(self.device)

                new_log_probs, entropy_sum = self._compute_new_log_probs_and_entropy(
                    obs_b, actions_q_per_branch, agent_indices
                )

                ratio = torch.exp(new_log_probs - old_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy_sum / self.n

                values_pred = self.critic(obs_b).squeeze()
                value_loss = nn.functional.mse_loss(values_pred, returns_b)

                loss = policy_loss + self.value_loss_coef * value_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actors.parameters()) + list(self.encoder.parameters()),
                    self.max_grad_norm
                )
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_sum / self.n
                n_updates += 1

        self.buffer.clear()

        if n_updates == 0:
            return {"loss": 0.0}

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path: str):
        state = {
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "use_gnn": self.use_gnn,
        }
        if self.encoder is not None:
            state["encoder"] = self.encoder.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(checkpoint["actors"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "encoder" in checkpoint and self.encoder is not None:
            self.encoder.load_state_dict(checkpoint["encoder"])
