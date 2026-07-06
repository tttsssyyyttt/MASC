"""IQL (Independent Q-Learning) Agent with BDQ branching and optional GNN encoder.

BDQ (Branching Dueling Q-Network) design:
- Each agent has n_upstream branches (1 for root nodes)
- Each branch outputs Q-values for [0, max_order]
- Action = argmax per branch → (q_up0, q_up1, ...)
- No AlphaNetwork needed — allocation is implicitly learned by per-upstream Q-values
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..base import MADRLAgent
from core.config import EnvConfig
from core.network import build_network
from models.encoders import make_encoder
from .buffer import ReplayBuffer


class BranchingQNetwork(nn.Module):
    """BDQ: shared trunk + per-upstream branches + MC Dropout.

    Dropout is kept active during inference for MC Dropout uncertainty estimation.
    """

    def __init__(self, input_dim: int, n_branches: int, max_order: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.n_branches = max(n_branches, 1)
        self.dropout_rate = dropout

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # MC Dropout for OOD detection
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.branches = nn.ModuleList([
            nn.Linear(hidden_dim, max_order + 1)
            for _ in range(self.n_branches)
        ])

    def forward(self, obs: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of (batch, max_order+1) Q-values per branch.

        Always uses dropout (even in eval mode) for MC Dropout.
        """
        features = self.trunk(obs)
        return [branch(features) for branch in self.branches]


class IQLAgent(MADRLAgent):
    """Independent Q-Learning agent with BDQ branching and optional GNN encoder."""

    def __init__(
        self,
        cfg: EnvConfig,
        obs_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 5000,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 200,
        device: str = "cpu",
        use_gnn: bool = False,
        edge_index: Optional[torch.Tensor] = None,
        layer_ids: Optional[torch.Tensor] = None,
        n_supply_layers: Optional[int] = None,
        encoder_update_freq: int = 4,
    ):
        self.cfg = cfg
        self.n = cfg.num_nodes
        self.max_order = cfg.max_order
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.use_gnn = use_gnn
        self.obs_dim = obs_dim

        self.upstream, self.downstream, self.topo_order, self.terminals = build_network(cfg)
        self.n_upstream = [len(self.upstream[i]) for i in range(self.n)]
        # self.n_action_branches = [1 if k == 0 else k + 1 for k in self.n_upstream]
        self.n_action_branches = [1 if k <= 1 else k + 1 for k in self.n_upstream]

        # Encoder: always use GRU→MLP (or GRU→GAT if use_gnn=True)
        self.encoder = make_encoder(
            obs_dim, hidden_dim, use_gnn=use_gnn,
            edge_index=edge_index, layer_ids=layer_ids,
            n_supply_layers=n_supply_layers or len(set(cfg.layers)),
            lead_time=cfg.lead_time,
        ).to(self.device)
        encoder_dim = hidden_dim

        # Branching Q-networks (one per agent)
        self.q_nets = nn.ModuleList([
            BranchingQNetwork(encoder_dim, self.n_action_branches[i], cfg.max_order, hidden_dim).to(self.device)
            for i in range(self.n)
        ])
        self.target_nets = nn.ModuleList([
            BranchingQNetwork(encoder_dim, self.n_action_branches[i], cfg.max_order, hidden_dim).to(self.device)
            for i in range(self.n)
        ])

        # Initialize target nets
        for i in range(self.n):
            self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())

        all_params = list(self.q_nets.parameters())
        if self.encoder is not None:
            all_params += list(self.encoder.parameters())
        self.optimizer = optim.Adam(all_params, lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0
        self.encoder_update_freq = encoder_update_freq

    def _get_epsilon(self) -> float:
        ratio = min(self.total_steps / self.epsilon_decay, 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio

    def _encode_obs(self, obs: np.ndarray) -> torch.Tensor:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if self.encoder is not None:
            with torch.no_grad():
                encoded = self.encoder(obs_t)
            return encoded.squeeze(0)
        return obs_t.squeeze(0)

    def _encode_obs_batch(self, obs: np.ndarray, with_grad: bool = False) -> torch.Tensor:
        obs_t = torch.FloatTensor(obs).to(self.device)
        if self.encoder is not None:
            if with_grad:
                encoded = self.encoder(obs_t)
            else:
                with torch.no_grad():
                    encoded = self.encoder(obs_t)
            return encoded
        return obs_t

    def get_actions(self, obs: np.ndarray, deterministic: bool = False) -> list:
        """Get BDQ actions: per-upstream order quantities.

        Returns:
            actions: list where actions[i] is:
                - int q for root nodes (no upstream)
                - dict {"q": q_total, "alpha": [w_up0, ...]} for non-root nodes
        """
        epsilon = 0.0 if deterministic else self._get_epsilon()
        actions = []
        encoded = self._encode_obs(obs)
        old_modes = [q_net.training for q_net in self.q_nets]
        if deterministic:
            for q_net in self.q_nets:
                q_net.eval()

        try:
            for i in range(self.n):
                k = self.n_upstream[i]
                if np.random.random() < epsilon:
                    # Random exploration
                    # if k == 0:
                    #     action_i = np.random.randint(0, self.max_order + 1)
                    # else:
                    #     action_i = {
                    #         "q": int(np.random.randint(0, self.max_order + 1)),
                    #         "alpha": [int(np.random.randint(0, self.max_order + 1)) for _ in range(k)],
                    #     }
                    if k == 0:
                        action_i = np.random.randint(0, self.max_order + 1)
                    elif k == 1:
                        action_i = [int(np.random.randint(0, self.max_order + 1))]
                    else:
                        action_i = {
                            "q": int(np.random.randint(0, self.max_order + 1)),
                            "alpha": [int(np.random.randint(0, self.max_order + 1)) for _ in range(k)],
                        }
                else:
                    with torch.no_grad():
                        q_branches = self.q_nets[i](encoded[i].unsqueeze(0))
                    # argmax per branch
                    action_vals = [int(q_b.argmax(dim=1).item()) for q_b in q_branches]
                    # if k == 0:
                    #     action_i = action_vals[0]  # single int for root nodes
                    # else:
                    #     action_i = {"q": action_vals[0], "alpha": action_vals[1:]}
                    if k == 0:
                        action_i = action_vals[0]
                    elif k == 1:
                        action_i = [action_vals[0]]
                    else:
                        action_i = {"q": action_vals[0], "alpha": action_vals[1:]}

                actions.append(action_i)
        finally:
            if deterministic:
                for q_net, was_training in zip(self.q_nets, old_modes):
                    q_net.train(was_training)

        return actions

    def get_entropy(self, obs: np.ndarray, tau: float = 1.0) -> np.ndarray:
        """Compute per-node entropy from softmax over Q-values.

        Args:
            obs: (N, obs_dim) observations
            tau: softmax temperature

        Returns:
            entropy: (N,) per-node total entropy (sum across branches)
        """
        encoded = self._encode_obs(obs)
        entropies = np.zeros(self.n)

        with torch.no_grad():
            for i in range(self.n):
                q_branches = self.q_nets[i](encoded[i].unsqueeze(0))
                node_entropy = 0.0
                for q_b in q_branches:
                    probs = torch.softmax(q_b / tau, dim=-1)
                    h = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                    node_entropy += h.item()
                entropies[i] = node_entropy

        return entropies

    def estimate_uncertainty(self, obs: np.ndarray, K: int = 5) -> float:
        """Estimate epistemic uncertainty via MC Dropout (Gal & Ghahramani, 2016).

        Runs K forward passes with dropout active. High variance in Q-values
        for the greedy action indicates the state is outside the training
        distribution (OOD).

        Args:
            obs: (N, obs_dim) observations
            K: number of MC samples

        Returns:
            mean_variance: average Q-value variance across agents and branches
        """
        encoded = self._encode_obs(obs)

        # Enable dropout for all Q-nets
        for q_net in self.q_nets:
            q_net.train()

        # Track max-Q values per agent/branch across K passes
        all_q_max = [[[0.0] * K for _ in range(b)] for b in self.n_action_branches]

        with torch.no_grad():
            for k in range(K):
                for i in range(self.n):
                    q_branches = self.q_nets[i](encoded[i].unsqueeze(0))
                    for j, q_b in enumerate(q_branches):
                        all_q_max[i][j][k] = q_b.max().item()

        # Restore eval mode
        for q_net in self.q_nets:
            q_net.eval()

        # Compute mean variance across agents
        variances = []
        for i in range(self.n):
            for j in range(len(all_q_max[i])):
                variances.append(np.var(all_q_max[i][j]))

        return float(np.mean(variances)) if variances else 0.0

    def store_transition(self, obs, actions, rewards, next_obs, dones):
        self.buffer.push(obs, actions, rewards, next_obs, dones)
        self.total_steps += 1

    def _branch_action_value(self, action, branch_idx: int, n_upstream: int) -> int:
        if n_upstream == 0:
            if isinstance(action, dict):
                return int(action.get("q", 0))
            return int(action)
        if isinstance(action, dict):
            if branch_idx == 0:
                return int(action.get("q", 0))
            alpha = action.get("alpha", [])
            return int(alpha[branch_idx - 1]) if branch_idx - 1 < len(alpha) else 0
        if branch_idx == 0:
            return int(min(sum(action), self.max_order))
        return int(action[branch_idx - 1]) if branch_idx - 1 < len(action) else 0

    def update(self, batch: Optional[dict] = None) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0}

        batch = self.buffer.sample(self.batch_size)

        update_encoder = (self.encoder is not None and
                         self.total_steps % self.encoder_update_freq == 0)

        if self.encoder is not None:
            encoded_obs = self._encode_obs_batch(batch["obs"], with_grad=update_encoder)
            encoded_next = self._encode_obs_batch(batch["next_obs"], with_grad=False)
        else:
            encoded_obs = torch.FloatTensor(batch["obs"]).to(self.device)
            encoded_next = torch.FloatTensor(batch["next_obs"]).to(self.device)

        # BDQ update: per-agent, per-branch Q-learning
        total_loss_tensor = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_branches_total = 0

        for i in range(self.n):
            obs_i = encoded_obs[:, i, :]
            next_obs_i = encoded_next[:, i, :]
            rewards_i = torch.FloatTensor(batch["rewards"][:, i]).to(self.device)
            dones_i = torch.FloatTensor(batch["dones"][:, i]).to(self.device)

            # Current Q-values for all branches
            q_branches = self.q_nets[i](obs_i)

            # Target Q-values (no grad)
            with torch.no_grad():
                next_q_branches = self.q_nets[i](next_obs_i)
                target_q_branches = self.target_nets[i](next_obs_i)

            # Per-branch TD loss
            for j, (q_b, target_q_b) in enumerate(zip(q_branches, target_q_branches)):
                # Action for this branch
                k = self.n_upstream[i]
                actions_j = torch.LongTensor([
                    self._branch_action_value(t[i], j, k)
                    for t in batch["actions"]
                ]).to(self.device)

                q_current = q_b.gather(1, actions_j.unsqueeze(1)).squeeze(1)
                # Double DQN target: online net selects, target net evaluates.
                next_actions = next_q_branches[j].argmax(dim=1, keepdim=True)
                next_q_max = target_q_b.gather(1, next_actions).squeeze(1)
                q_target = rewards_i + self.gamma * next_q_max * (1 - dones_i)

                # Huber loss is less sensitive to occasional large TD errors.
                loss = nn.functional.smooth_l1_loss(q_current, q_target)
                total_loss_tensor = total_loss_tensor + loss
                n_branches_total += 1

        if n_branches_total > 0:
            total_loss_tensor = total_loss_tensor / n_branches_total

        self.optimizer.zero_grad()
        total_loss_tensor.backward()
        nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 5)
        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            for i in range(self.n):
                self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())

        return {"loss": total_loss_tensor.item()}

    def save(self, path: str):
        state = {
            "q_nets": self.q_nets.state_dict(),
            "target_nets": self.target_nets.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "use_gnn": self.use_gnn,
        }
        if self.encoder is not None:
            state["encoder"] = self.encoder.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_nets.load_state_dict(checkpoint["q_nets"])
        self.target_nets.load_state_dict(checkpoint["target_nets"])
        if "encoder" in checkpoint and self.encoder is not None:
            self.encoder.load_state_dict(checkpoint["encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
