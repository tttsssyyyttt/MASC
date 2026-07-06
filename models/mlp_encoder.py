"""MLP encoder — fallback when use_gnn=False."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Simple MLP encoder for non-GNN mode.

    Input: temporal_feat (batch, N, input_dim)
    Output: feat (batch, N, hidden_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
