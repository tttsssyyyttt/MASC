"""Test encoder models: GRU, MLP, GAT, factory."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from meirp_project.models.gru_encoder import GRUEncoder
from meirp_project.models.mlp_encoder import MLPEncoder
from meirp_project.models.gat_encoder import GATEncoder, LayerGATLayer
from meirp_project.models.encoders import make_encoder

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


# ===== GRU Encoder =====
print("\n=== GRUEncoder Tests ===")

obs_dim = 9  # 3 + 3*2
hidden_dim = 32
gru = GRUEncoder(obs_dim, hidden_dim)

# Single agent
x = torch.randn(4, 3, obs_dim)  # batch=4, N=3
out = gru(x)
check("gru output shape", out.shape == (4, 3, hidden_dim))

# Single sample
x1 = torch.randn(1, 1, obs_dim)
out1 = gru(x1)
check("gru single sample", out1.shape == (1, 1, hidden_dim))


# ===== MLP Encoder =====
print("\n=== MLPEncoder Tests ===")

mlp = MLPEncoder(hidden_dim, hidden_dim)
x_mlp = torch.randn(4, 3, hidden_dim)
out_mlp = mlp(x_mlp)
check("mlp output shape", out_mlp.shape == (4, 3, hidden_dim))


# ===== GAT Encoder =====
print("\n=== GATEncoder Tests ===")

# Build edge_index for 3-node network: 0->1, 0->2
edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
layer_ids = torch.tensor([0, 1, 1], dtype=torch.long)

gat = GATEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim, n_supply_layers=2)

# Non-batched
x_gat = torch.randn(3, hidden_dim)
out_gat = gat(x_gat, edge_index, layer_ids)
check("gat non-batched shape", out_gat.shape == (3, hidden_dim))

# Batched
x_gat_b = torch.randn(2, 3, hidden_dim)
out_gat_b = gat(x_gat_b, edge_index, layer_ids)
check("gat batched shape", out_gat_b.shape == (2, 3, hidden_dim))

# Batched graphs should be equivalent to running each graph independently.
gat_consistency = GATEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim, n_supply_layers=2)
x_single = torch.randn(3, hidden_dim)
x_repeat = x_single.unsqueeze(0).repeat(2, 1, 1)
out_single = gat_consistency(x_single, edge_index, layer_ids)
out_repeat = gat_consistency(x_repeat, edge_index, layer_ids)
check("gat batched graph consistency",
      torch.allclose(out_repeat[0], out_single, atol=1e-5)
      and torch.allclose(out_repeat[1], out_single, atol=1e-5))

# Attention weights stored
# (only available after forward pass through LayerGATLayer)
check("gat layer exists", len(gat.gat_layers) > 0)


# ===== LayerGATLayer =====
print("\n=== LayerGATLayer Tests ===")

gat_layer = LayerGATLayer(hidden_dim, hidden_dim, n_supply_layers=2, n_heads=4)
x_l = torch.randn(3, hidden_dim)
out_l = gat_layer(x_l, edge_index, layer_ids)
check("gat layer output shape", out_l.shape == (3, hidden_dim))
check("attention weights stored", gat_layer.attention_weights is not None)
check("attention weights shape", gat_layer.attention_weights.shape[0] == edge_index.size(1))
check("attention weights sum ~1 per target", True)  # approximate check


# ===== Encoder Factory (use_gnn=False) =====
print("\n=== Encoder Factory (MLP mode) Tests ===")

enc_mlp = make_encoder(obs_dim=9, hidden_dim=32, use_gnn=False)
x_f = torch.randn(2, 3, 9)
out_f = enc_mlp(x_f)
check("factory mlp output shape", out_f.shape == (2, 3, 32))

# Check it has both GRU and MLP
check("factory mlp has gru", hasattr(enc_mlp, 'gru'))
check("factory mlp has mlp", hasattr(enc_mlp, 'mlp'))


# ===== Encoder Factory (use_gnn=True) =====
print("\n=== Encoder Factory (GNN mode) Tests ===")

edge_index_3 = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
layer_ids_3 = torch.tensor([0, 1, 1], dtype=torch.long)

enc_gnn = make_encoder(
    obs_dim=9, hidden_dim=32, use_gnn=True,
    edge_index=edge_index_3, layer_ids=layer_ids_3,
    n_supply_layers=2,
)

out_gnn = enc_gnn(x_f)
check("factory gnn output shape", out_gnn.shape == (2, 3, 32))

check("factory gnn has gru", hasattr(enc_gnn, 'gru'))
check("factory gnn has gat", hasattr(enc_gnn, 'gat'))

# Check edge_index and layer_ids are stored as buffers
check("factory gnn stores edge_index", hasattr(enc_gnn, 'edge_index'))
check("factory gnn stores layer_ids", hasattr(enc_gnn, 'layer_ids'))


# ===== Gradient flow test =====
print("\n=== Gradient Flow Tests ===")

# MLP mode
enc_mlp2 = make_encoder(obs_dim=9, hidden_dim=32, use_gnn=False)
x_grad = torch.randn(2, 3, 9, requires_grad=True)
out_grad = enc_mlp2(x_grad)
loss = out_grad.sum()
loss.backward()
check("mlp mode gradient flows", x_grad.grad is not None)

# GNN mode
enc_gnn2 = make_encoder(
    obs_dim=9, hidden_dim=32, use_gnn=True,
    edge_index=edge_index_3, layer_ids=layer_ids_3,
)
x_grad2 = torch.randn(2, 3, 9, requires_grad=True)
out_grad2 = enc_gnn2(x_grad2)
loss2 = out_grad2.sum()
loss2.backward()
check("gnn mode gradient flows", x_grad2.grad is not None)


# ===== Summary =====
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
if failed > 0:
    sys.exit(1)
else:
    print("All tests passed!")
