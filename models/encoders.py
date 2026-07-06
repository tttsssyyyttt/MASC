"""Encoder factory — creates the right encoder based on use_gnn flag."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .gru_encoder import GRUEncoder
from .gat_encoder import GATEncoder
from .mlp_encoder import MLPEncoder


def make_encoder(
    obs_dim: int,
    hidden_dim: int = 64,
    use_gnn: bool = False,
    edge_index=None,
    layer_ids=None,
    n_supply_layers: int = 4,
    n_heads: int = 4,
    n_graph_layers: int = 2,
    dropout: float = 0.0,
    lead_time: int = 2,
) -> nn.Module:
    """Create encoder pipeline: GRU -> (GAT | MLP).

    Args:
        obs_dim: observation dimension per agent
        hidden_dim: output hidden dimension
        use_gnn: if True, use GAT after GRU; else use MLP after GRU
        edge_index: required if use_gnn=True, shape (2, E)
        layer_ids: required if use_gnn=True, shape (N,)
        n_supply_layers: number of supply chain layers (for layer embedding)
        n_heads: number of attention heads
        n_graph_layers: number of GAT layers
        dropout: dropout rate
        lead_time: supply chain lead time (determines GRU sequence length)

    Returns:
        nn.Module that maps obs -> hidden features
    """
    gru = GRUEncoder(obs_dim, hidden_dim, lead_time=lead_time)

    if use_gnn:
        if edge_index is None or layer_ids is None:
            raise ValueError("edge_index and layer_ids required when use_gnn=True")
        gat = GATEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers_graph=n_graph_layers,
            n_supply_layers=n_supply_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        return _GRUGNNEncoder(gru, gat, edge_index, layer_ids)
    else:
        mlp = MLPEncoder(hidden_dim, hidden_dim)
        return _GRUMLPEncoder(gru, mlp)


class _GRUMLPEncoder(nn.Module):
    """Pipeline: GRU -> MLP."""

    def __init__(self, gru: GRUEncoder, mlp: MLPEncoder):
        super().__init__()
        self.gru = gru
        self.mlp = mlp

    def forward(self, obs):
        """obs: (batch, N, obs_dim) -> (batch, N, hidden_dim)"""
        h = self.gru(obs)
        return self.mlp(h)


class _GRUGNNEncoder(nn.Module):
    """Pipeline: GRU -> GAT."""

    def __init__(self, gru: GRUEncoder, gat: GATEncoder, edge_index, layer_ids):
        super().__init__()
        self.gru = gru
        self.gat = gat
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("layer_ids", layer_ids)

    def forward(self, obs):
        """obs: (batch, N, obs_dim) -> (batch, N, hidden_dim)"""
        h = self.gru(obs)
        return self.gat(h, self.edge_index, self.layer_ids)
