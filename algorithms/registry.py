"""Algorithm registry — create agents by name with optional GNN support."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import MADRLAgent
from .iql.agent import IQLAgent
from .qmix.agent import QMIXAgent
from .mappo.agent import MAPPOAgent
from core.config import EnvConfig
from core.network import build_network


def make_agent(
    name: str,
    cfg: EnvConfig,
    obs_dim: Optional[int] = None,
    device: str = "cpu",
    use_gnn: bool = False,
    **kwargs,
) -> MADRLAgent:
    """Create an agent by algorithm name.

    Args:
        name: one of "iql", "qmix", "mappo"
        cfg: environment configuration
        obs_dim: observation dimension (auto-computed if None)
        device: torch device
        use_gnn: if True, use GAT encoder in the agent
        **kwargs: additional algorithm-specific hyperparameters

    Returns:
        MADRLAgent instance
    """
    if obs_dim is None:
        obs_dim = 6 + 3 * cfg.lead_time

    name = name.lower()

    # Build graph info if use_gnn
    edge_index = None
    layer_ids = None
    n_supply_layers = None
    if use_gnn:
        upstream, downstream, topo_order, terminals = build_network(cfg)
        # Build edge_index from cfg.edges
        src_list = [e[0] for e in cfg.edges]
        dst_list = [e[1] for e in cfg.edges]
        # Add reverse edges for undirected message passing
        all_src = src_list + dst_list
        all_dst = dst_list + src_list
        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
        layer_ids = torch.tensor(cfg.layers, dtype=torch.long)
        n_supply_layers = len(set(cfg.layers))

    if name == "iql":
        iql_kwargs = {k: v for k, v in kwargs.items() if k in (
            "hidden_dim", "lr", "gamma", "epsilon_start", "epsilon_end",
            "epsilon_decay", "buffer_capacity", "batch_size",
            "target_update_freq", "encoder_update_freq",
        )}
        return IQLAgent(
            cfg=cfg, obs_dim=obs_dim, device=device,
            use_gnn=use_gnn, edge_index=edge_index,
            layer_ids=layer_ids, n_supply_layers=n_supply_layers,
            **iql_kwargs,
        )
    elif name == "qmix":
        qmix_kwargs = {k: v for k, v in kwargs.items() if k in (
            "hidden_dim", "mixer_hidden", "lr", "gamma",
            "epsilon_start", "epsilon_end", "epsilon_decay",
            "buffer_capacity", "batch_size", "target_update_freq",
            "encoder_update_freq",
        )}
        return QMIXAgent(
            cfg=cfg, obs_dim=obs_dim, device=device,
            use_gnn=use_gnn, edge_index=edge_index,
            layer_ids=layer_ids, n_supply_layers=n_supply_layers,
            **qmix_kwargs,
        )
    elif name == "mappo":
        mappo_kwargs = {k: v for k, v in kwargs.items() if k in (
            "hidden_dim", "lr_actor", "lr_critic", "gamma",
            "gae_lambda", "clip_epsilon", "epochs_per_update",
            "mini_batch_size", "entropy_coef", "value_loss_coef",
            "max_grad_norm",
        )}
        return MAPPOAgent(
            cfg=cfg, obs_dim=obs_dim, device=device,
            use_gnn=use_gnn, edge_index=edge_index,
            layer_ids=layer_ids, n_supply_layers=n_supply_layers,
            **mappo_kwargs,
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}. Choose from: iql, qmix, mappo")
