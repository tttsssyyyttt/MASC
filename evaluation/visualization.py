"""Visualization utilities for evaluation results.

Includes: training curves, cost breakdown, comparison tables,
GNN attention heatmaps, network topology visualization.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def plot_training_curve(rewards: List[float], title: str = "Training Curve", save_path: Optional[str] = None):
    """Plot training reward curve."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, label="Raw")
    
    # Moving average
    window = min(50, len(rewards) // 5) if len(rewards) > 10 else len(rewards)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed, label=f"MA-{window}")
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_breakdown(metrics: dict, title: str = "Cost Breakdown", save_path: Optional[str] = None):
    """Plot cost breakdown as bar chart."""
    import matplotlib.pyplot as plt

    holding = metrics.get("total_holding", 0)
    backlog = metrics.get("total_backlog", 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Holding", "Backlog"], [holding, backlog], color=["steelblue", "coral"])
    ax.set_ylabel("Total Cost")
    ax.set_title(title)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.0f}", ha="center", va="bottom")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_table(results: Dict[str, dict], save_path: Optional[str] = None):
    """Plot algorithm comparison as table."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    headers = ["Algorithm", "Cost/Step", "Fill Rate", "Holding", "Backlog"]
    rows = []
    for name, m in results.items():
        rows.append([
            name,
            f"{m.get('avg_cost_per_step', 0):.1f}",
            f"{m.get('avg_fill_rate', 0):.3f}",
            f"{m.get('total_holding', 0):.0f}",
            f"{m.get('total_backlog', 0):.0f}",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    edge_index: np.ndarray,
    node_labels: Optional[List[str]] = None,
    title: str = "GNN Attention Weights",
    save_path: Optional[str] = None,
):
    """Plot GNN attention weights as a heatmap.

    Args:
        attention_weights: (E,) attention weight per edge
        edge_index: (2, E) edge source/target indices
        node_labels: optional labels for nodes
        title: plot title
        save_path: path to save figure
    """
    import matplotlib.pyplot as plt

    N = int(max(edge_index[0].max(), edge_index[1].max())) + 1
    E = len(attention_weights)

    # Build attention matrix
    attn_matrix = np.zeros((N, N))
    for e in range(E):
        src, dst = int(edge_index[0, e]), int(edge_index[1, e])
        attn_matrix[dst, src] = attention_weights[e]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_matrix, cmap="YlOrRd", aspect="auto")

    if node_labels is None:
        node_labels = [str(i) for i in range(N)]

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(node_labels, rotation=45, ha="right")
    ax.set_yticklabels(node_labels)
    ax.set_xlabel("Source Node")
    ax.set_ylabel("Target Node")
    ax.set_title(title)

    for i in range(N):
        for j in range(N):
            val = attn_matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.5 else "black", fontsize=8)

    plt.colorbar(im, ax=ax, label="Attention Weight")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_network_topology(
    edges,
    layers: List[int],
    node_labels: Optional[List[str]] = None,
    attention_weights: Optional[np.ndarray] = None,
    edge_index: Optional[np.ndarray] = None,
    title: str = "Supply Chain Network",
    save_path: Optional[str] = None,
):
    """Plot supply chain network topology with optional attention-weighted edges."""
    import matplotlib.pyplot as plt

    N = len(layers)
    n_layers = max(layers) + 1

    layer_counts = [layers.count(l) for l in range(n_layers)]
    pos = {}
    layer_idx = [0] * n_layers

    for i in range(N):
        l = layers[i]
        y_offset = (layer_idx[l] - (layer_counts[l] - 1) / 2) * 1.5
        pos[i] = (l, y_offset)
        layer_idx[l] += 1

    fig, ax = plt.subplots(figsize=(10, 6))

    edge_widths = None
    if attention_weights is not None and edge_index is not None:
        attn_dict = {}
        for e in range(attention_weights.shape[0]):
            src, dst = int(edge_index[0, e]), int(edge_index[1, e])
            attn_dict[(src, dst)] = attention_weights[e]
        edge_widths = []
        for src, dst in edges:
            w = attn_dict.get((src, dst), 0.1)
            edge_widths.append(max(0.5, w * 5))

    for idx, (src, dst) in enumerate(edges):
        x = [pos[src][0], pos[dst][0]]
        y = [pos[src][1], pos[dst][1]]
        lw = edge_widths[idx] if edge_widths else 1.0
        color = "steelblue" if edge_widths is None else plt.cm.YlOrRd(
            min(1.0, edge_widths[idx] / 5)
        )
        ax.plot(x, y, color=color, linewidth=lw, alpha=0.7, zorder=1)

    layer_colors = plt.cm.Set2(np.linspace(0, 1, n_layers))
    for i in range(N):
        l = layers[i]
        ax.scatter(pos[i][0], pos[i][1], s=200, c=[layer_colors[l]],
                   edgecolors="black", zorder=2)
        label = node_labels[i] if node_labels else str(i)
        ax.annotate(label, pos[i], textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("Layer")
    ax.set_title(title)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"Layer {l}" for l in range(n_layers)])
    ax.grid(True, alpha=0.2)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
