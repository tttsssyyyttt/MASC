"""GRU temporal encoder — extracts time-series features from obs history.

Observation structure (obs_dim = 3 + 3L, where L = lead_time):
    [0]:       inventory (current)
    [1]:       backlog   (current)
    [2]:       demand    (current)
    [3:3+L]:   pipeline  (L steps: t-1, t-2, ...)
    [3+L:3+2L]: demand_hist (L steps)
    [3+2L:3+3L]: order_hist (L steps)

We reshape into a sequence of length (L+1):
    t=0 (current): [inventory, backlog, demand]
    t=-1:          [pipeline[0], demand_hist[0], order_hist[0]]
    t=-2:          [pipeline[1], demand_hist[1], order_hist[1]]
    ...

This gives the GRU a meaningful temporal sequence to model supply chain dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """GRU-based temporal encoder with proper multi-step sequence processing."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64, lead_time: int = 2, num_layers: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.lead_time = lead_time
        self.seq_len = lead_time + 1  # current + L history steps
        self.feat_per_step = 3  # inventory/backlog/demand or pipeline/demand_hist/order_hist

        # Project each time-step feature (3 dims) to hidden_dim before GRU
        self.input_proj = nn.Linear(self.feat_per_step, hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations as temporal sequence through GRU.

        Args:
            obs: (batch, N, obs_dim) where obs_dim = 3 + 3*L

        Returns:
            temporal_feat: (batch, N, hidden_dim)
        """
        batch_size, N, _ = obs.shape
        L = self.lead_time

        # Split obs into time steps: (batch, N, L+1, 3)
        # Current step: obs[:, :, 0:3]
        current = obs[:, :, :3].unsqueeze(2)  # (batch, N, 1, 3)

        # History steps: reshape pipeline, demand_hist, order_hist
        pipeline = obs[:, :, 3:3+L]         # (batch, N, L)
        demand_hist = obs[:, :, 3+L:3+2*L]  # (batch, N, L)
        order_hist = obs[:, :, 3+2*L:3+3*L] # (batch, N, L)

        # Stack as features per time step: (batch, N, L, 3)
        history = torch.stack([pipeline, demand_hist, order_hist], dim=-1)

        # Concatenate: current first, then history in reverse (most recent first)
        # (batch, N, L+1, 3)
        sequence = torch.cat([current, history], dim=2)

        # Reshape for GRU: (batch * N, L+1, hidden_dim)
        x = sequence.reshape(batch_size * N, self.seq_len, self.feat_per_step)
        x = self.input_proj(x)  # (batch*N, seq_len, hidden_dim)

        output, _ = self.gru(x)
        # Take last hidden state as summary
        feat = output[:, -1, :]  # (batch * N, hidden_dim)

        return feat.reshape(batch_size, N, self.hidden_dim)
