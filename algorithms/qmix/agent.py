"""QMIX Agent with BDQ branching and optional GNN encoder.

Uses BranchingQNetwork from IQL for per-agent Q-values.
QMIX mixer takes the sum of per-agent chosen Q-values across branches.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..base import MADRLAgent
from ..iql.agent import BranchingQNetwork
from ..iql.buffer import ReplayBuffer
from core.config import EnvConfig
from core.network import build_network
from models.encoders import make_encoder
from .mixer import QMixer


class QMIXAgent(MADRLAgent):
    """QMIX agent with BDQ branching, monotonic value factorization, and optional GNN encoder."""

    def __init__(
        self,
        cfg: EnvConfig,
        obs_dim: int,
        hidden_dim: int = 128,
        mixer_hidden: int = 32,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        buffer_capacity: int = 50000,
        batch_size: int = 32,
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
        self.n_action_branches = [1 if k <= 1 else k + 1 for k in self.n_upstream]

        # Encoder: always use GRU→MLP (same as IQL)
        self.encoder = make_encoder(
            obs_dim, hidden_dim, use_gnn=use_gnn,
            edge_index=edge_index, layer_ids=layer_ids,
            n_supply_layers=n_supply_layers or len(set(cfg.layers)),
            lead_time=cfg.lead_time,
        ).to(self.device)
        encoder_dim = hidden_dim

        self.state_dim = self.n * encoder_dim

        # Branching Q-networks
        self.q_nets = nn.ModuleList([
            BranchingQNetwork(encoder_dim, self.n_action_branches[i], cfg.max_order, hidden_dim).to(self.device)
            for i in range(self.n)
        ])
        self.target_q_nets = nn.ModuleList([
            BranchingQNetwork(encoder_dim, self.n_action_branches[i], cfg.max_order, hidden_dim).to(self.device)
            for i in range(self.n)
        ])
        for i in range(self.n):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())

        # Mixer
        self.mixer = QMixer(self.n, self.state_dim, mixer_hidden).to(self.device)
        self.target_mixer = QMixer(self.n, self.state_dim, mixer_hidden).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        all_params = (
            list(self.q_nets.parameters())
            + list(self.mixer.parameters())
        )
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
        with torch.no_grad():
            encoded = self.encoder(obs_t)
        return encoded.squeeze(0)

    def _encode_obs_batch(self, obs: np.ndarray, with_grad: bool = False) -> torch.Tensor:
        obs_t = torch.FloatTensor(obs).to(self.device)
        if with_grad:
            encoded = self.encoder(obs_t)
        else:
            with torch.no_grad():
                encoded = self.encoder(obs_t)
        return encoded

    def get_actions(self, obs: np.ndarray, deterministic: bool = False) -> list:
        """Get BDQ actions: per-upstream order quantities."""
        epsilon = 0.0 if deterministic else self._get_epsilon()
        actions = []
        encoded = self._encode_obs(obs)

        # for i in range(self.n):
        #     k = self.n_upstream[i]
        #     if np.random.random() < epsilon:
        #         # if k == 0:
        #         #     action_i = np.random.randint(0, self.max_order + 1)
        #         # else:
        #         #     action_i = {
        #         #         "q": int(np.random.randint(0, self.max_order + 1)),
        #         #         "alpha": [int(np.random.randint(0, self.max_order + 1)) for _ in range(k)],
        #         #     }
        #         if k == 0:
        #             action_i = np.random.randint(0, self.max_order + 1)
        #         elif k == 1:
        #             action_i = [int(np.random.randint(0, self.max_order + 1))]
        #         else:
        #             action_i = {
        #                 "q": int(np.random.randint(0, self.max_order + 1)),
        #                 "alpha": [int(np.random.randint(0, self.max_order + 1)) for _ in range(k)],
        #             }
        #     else:
        #         with torch.no_grad():
        #             q_branches = self.q_nets[i](encoded[i].unsqueeze(0))
        #         action_vals = [int(q_b.argmax(dim=1).item()) for q_b in q_branches]
        #         # if k == 0:
        #         #     action_i = action_vals[0]
        #         # else:
        #         #     action_i = {"q": action_vals[0], "alpha": action_vals[1:]}
        #         if k == 0:
        #             action_i = action_vals[0]
        #         elif k == 1:
        #             action_i = [action_vals[0]]
        #         else:
        #             action_i = {"q": action_vals[0], "alpha": action_vals[1:]}
        #
        #     actions.append(action_i)
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
        """Compute per-node entropy from softmax over Q-values."""
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
        B = len(batch["obs"])

        if self.encoder is not None:
            update_encoder = self.total_steps % self.encoder_update_freq == 0
            encoded_obs = self._encode_obs_batch(batch["obs"], with_grad=update_encoder)
            encoded_next = self._encode_obs_batch(batch["next_obs"], with_grad=False)
        else:
            encoded_obs = torch.FloatTensor(batch["obs"]).to(self.device)
            encoded_next = torch.FloatTensor(batch["next_obs"]).to(self.device)

        # QMIX update: sum per-agent chosen Q across branches, then mix
        q_chosen_list = []
        q_next_max_list = []

        for i in range(self.n):
            obs_i = encoded_obs[:, i, :]
            next_obs_i = encoded_next[:, i, :]
            # Current Q-values
            q_branches = self.q_nets[i](obs_i)

            # Target Q-values
            with torch.no_grad():
                target_q_branches = self.target_q_nets[i](next_obs_i)

            # Sum Q-values across branches for QMIX input
            q_chosen_sum = torch.zeros(B, device=self.device)
            q_next_max_sum = torch.zeros(B, device=self.device)

            for j, (q_b, target_q_b) in enumerate(zip(q_branches, target_q_branches)):
                k = self.n_upstream[i]
                actions_j = torch.LongTensor([
                    self._branch_action_value(t[i], j, k)
                    for t in batch["actions"]
                ]).to(self.device)

                q_chosen_sum += q_b.gather(1, actions_j.unsqueeze(1)).squeeze(1)
                q_next_max_sum += target_q_b.max(dim=1)[0]
                q_chosen_agent = q_chosen_sum / max(1, len(q_branches))
                q_next_agent = q_next_max_sum / max(1, len(target_q_branches))

            # q_chosen_list.append(q_chosen_sum)
            # q_next_max_list.append(q_next_max_sum)
            q_chosen_list.append(q_chosen_agent)
            q_next_max_list.append(q_next_agent)

        q_chosen_t = torch.stack(q_chosen_list, dim=1)   # (B, n_agents)
        q_next_max_t = torch.stack(q_next_max_list, dim=1)

        states_t = encoded_obs.reshape(B, -1)
        next_states_t = encoded_next.reshape(B, -1)

        q_tot = self.mixer(q_chosen_t, states_t)
        with torch.no_grad():
            q_tot_next = self.target_mixer(q_next_max_t, next_states_t)

        rewards_t = torch.FloatTensor(batch["rewards"].sum(axis=1)).to(self.device)
        dones_t = torch.FloatTensor(batch["dones"][:, 0]).to(self.device)

        targets = rewards_t.unsqueeze(1) + self.gamma * q_tot_next * (1 - dones_t.unsqueeze(1))
        targets = targets.clamp(-20.0, 20.0)
        # qmix_loss = nn.functional.mse_loss(q_tot, targets.detach())
        qmix_loss = nn.functional.smooth_l1_loss(q_tot, targets.detach())

        self.optimizer.zero_grad()
        qmix_loss.backward()
        nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 1)
        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            for i in range(self.n):
                self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return {"loss": qmix_loss.item()}

    def save(self, path: str):
        state = {
            "q_nets": self.q_nets.state_dict(),
            "target_q_nets": self.target_q_nets.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
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
        self.target_q_nets.load_state_dict(checkpoint["target_q_nets"])
        if "encoder" in checkpoint and self.encoder is not None:
            self.encoder.load_state_dict(checkpoint["encoder"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
