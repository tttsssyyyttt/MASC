"""Layer-aware GAT encoder — graph attention with per-layer transformation matrices.

Key improvement: each supply chain layer l has its own linear transformation W_l,
so nodes at different echelons project their features differently before attention.

Attention: e_ij = LeakyReLU(a^T [W_{l_i} h_i || W_{l_j} h_j || layer_emb])

OPTIMIZED: Vectorized batch and per-layer operations — no Python for-loops.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerGATLayer(nn.Module):
    """Single GAT layer with layer-aware attention and per-layer W_l.

    OPTIMIZED: Per-layer transformation is done via einsum (no Python loop).
    - W_all: (n_supply_layers, out_dim * n_heads, in_dim) parameter tensor
    - Forward: W_selected = W_all[layer_ids] -> einsum -> (N, out_dim * n_heads)
    """

    def __init__(self, in_dim: int, out_dim: int, n_supply_layers: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.n_supply_layers = n_supply_layers

        # Vectorized per-layer transformation: (n_supply_layers, out_dim * n_heads, in_dim)
        self.W_all = nn.Parameter(torch.empty(n_supply_layers, out_dim * n_heads, in_dim))
        # Initialize same as nn.Linear default (kaiming_uniform_)
        for l in range(n_supply_layers):
            nn.init.kaiming_uniform_(self.W_all[l], a=math.sqrt(5))

        # Attention vector: takes [h_i || h_j || layer_emb] -> scalar
        self.a = nn.Parameter(torch.zeros(2 * out_dim + n_supply_layers))

        # Layer embedding
        self.layer_emb = nn.Embedding(n_supply_layers, n_supply_layers)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        # Store attention weights for visualization
        self.attention_weights = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edges [source, target]
            layer_ids: (N,) layer index per node

        Returns:
            out: (N, out_dim) — mean over heads
        """
        N = x.size(0)

        # Vectorized per-layer transformation via einsum (no Python loop)
        # W_selected: (N, out_dim * n_heads, in_dim), indexed by layer_ids
        W_selected = self.W_all[layer_ids]  # (N, out_dim * n_heads, in_dim)
        h = torch.einsum('ni,noi->no', x, W_selected)  # (N, out_dim * n_heads)
        h = h.view(N, self.n_heads, self.out_dim)

        # Layer embeddings
        l_emb = self.layer_emb(layer_ids)  # (N, n_supply_layers)

        # Compute attention for each edge
        src, dst = edge_index[0], edge_index[1]

        h_src = h[src]  # (E, heads, out_dim)
        h_dst = h[dst]  # (E, heads, out_dim)

        # Average over heads for attention computation
        h_src_mean = h_src.mean(dim=1)  # (E, out_dim)
        h_dst_mean = h_dst.mean(dim=1)  # (E, out_dim)

        # Layer embedding for source node
        l_src = l_emb[src]  # (E, n_supply_layers)

        # Attention: a^T [h_i || h_j || layer_emb]
        attn_input = torch.cat([h_src_mean, h_dst_mean, l_src], dim=-1)  # (E, 2*out + n_layers)
        e = self.leaky_relu(F.linear(attn_input, self.a.unsqueeze(0)).squeeze(-1))  # (E,)

        # Softmax per target node
        e_exp = torch.exp(e - e.max())
        e_sum = torch.zeros(N, device=x.device)
        e_sum.scatter_add_(0, dst, e_exp)
        alpha = e_exp / (e_sum[dst] + 1e-10)  # (E,)
        alpha = self.dropout(alpha)

        # Store for visualization
        self.attention_weights = alpha.detach()

        # Aggregate: weighted sum of source features per head
        out = torch.zeros(N, self.n_heads, self.out_dim, device=x.device)
        alpha_expanded = alpha.unsqueeze(1).unsqueeze(2)  # (E, 1, 1)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(h_src), h_src * alpha_expanded)

        # Mean over heads
        out = out.mean(dim=1)  # (N, out_dim)

        return out


class GATEncoder(nn.Module):
    """Multi-layer GAT encoder with layer-aware attention and per-layer W_l.

    Input: temporal_feat (batch, N, input_dim) + edge_index + layer_ids
    Output: graph_feat (batch, N, hidden_dim)

    OPTIMIZED: Batch processing is fully vectorized by combining all batch
    elements into a single large graph with offset node indices.
    No Python for-loop over batch dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers_graph: int = 2,
        n_supply_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for _ in range(n_layers_graph):
            self.gat_layers.append(
                LayerGATLayer(hidden_dim, hidden_dim, n_supply_layers, n_heads, dropout)
            )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, N, input_dim) or (N, input_dim)
            edge_index: (2, E)
            layer_ids: (N,)

        Returns:
            feat: (batch, N, hidden_dim) or (N, hidden_dim)
        """
        is_batched = x.dim() == 3

        if is_batched:
            batch_size, N, _ = x.shape

            # Flatten to (B*N, input_dim) and project
            h = self.input_proj(x.reshape(batch_size * N, -1))  # (B*N, hidden_dim)

            # Ensure edge_index and layer_ids are on the same device as x
            edge_index = edge_index.to(x.device)
            layer_ids = layer_ids.to(x.device)

            # Create batched edge_index with node offsets
            # Each batch b gets offset b*N added to its edge indices
            offsets = torch.arange(batch_size, device=x.device).unsqueeze(1) * N  # (B, 1)
            edge_index_batched = (
                edge_index.unsqueeze(0) + offsets.unsqueeze(2)
            ).permute(1, 0, 2).reshape(2, -1)  # (2, B*E)

            # Expand layer_ids for all batches: (B*N,)
            layer_ids_batched = layer_ids.unsqueeze(0).expand(batch_size, -1).reshape(-1)

            # Run GAT layers on the combined graph
            for gat_layer in self.gat_layers:
                h_res = gat_layer(h, edge_index_batched, layer_ids_batched)
                h = self.norm(h + h_res)

            return h.reshape(batch_size, N, -1)
        else:
            h = self.input_proj(x)
            for gat_layer in self.gat_layers:
                h_res = gat_layer(h, edge_index, layer_ids)
                h = self.norm(h + h_res)
            return h

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights from the last GAT layer for visualization.

        Returns:
            attention_weights: (E,) tensor from the last layer, or None
        """
        if len(self.gat_layers) > 0:
            return self.gat_layers[-1].attention_weights
        return None
