"""
GNN encoder + regression heads for the Mechanical MNIST energy/force
metamodel.

Architecture (deliberately structured in the same *spirit* as the original
GNN-PINN repo -- a custom edge-aware graph-conv layer followed by a
multi-statistic global readout -- but implemented from scratch as idiomatic
PyTorch Geometric `MessagePassing` modules, not a port):

    EdgeConditionedConv  : message-passing layer whose messages are a small
                            MLP over (x_i, x_j, edge_attr) -- lets the layer
                            use the physical relative-position/distance
                            between grid nodes, unlike a plain GCN.
    MultiStatReadout      : global pooling that concatenates mean / max / sum
                            / std node statistics into one graph embedding
                            (analogous role to the original's GRLayer).
    MechGNNPINN           : encoder + readout -> two heads (strain-energy
                            Psi, reaction-force F), both conditioned on the
                            scalar load-step `delta`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class EdgeConditionedConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.upd_mlp(torch.cat([x, aggr_out], dim=-1))


class MultiStatReadout(nn.Module):
    """Concatenates mean / max / sum / std node-feature statistics per graph."""

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean_ = scatter(x, batch, dim=0, reduce="mean")
        max_ = scatter(x, batch, dim=0, reduce="max")
        sum_ = scatter(x, batch, dim=0, reduce="sum")
        sq_mean = scatter(x * x, batch, dim=0, reduce="mean")
        std_ = torch.sqrt(torch.clamp(sq_mean - mean_ * mean_, min=1e-12))
        return torch.cat([mean_, max_, sum_, std_], dim=-1)


class MechGNNPINN(nn.Module):
    def __init__(self, node_dim: int = 5, edge_dim: int = 3, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Linear(node_dim, hidden)
        self.conv1 = EdgeConditionedConv(hidden, hidden, edge_dim)
        self.conv2 = EdgeConditionedConv(hidden, hidden, edge_dim)
        self.readout = MultiStatReadout()

        embed_dim = 4 * hidden  # mean/max/sum/std concat
        self.psi_head = nn.Sequential(
            nn.Linear(embed_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.force_head = nn.Sequential(
            nn.Linear(embed_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def embed(self, x, edge_index, edge_attr, batch):
        h = torch.relu(self.encoder(x))
        h = torch.relu(self.conv1(h, edge_index, edge_attr))
        h = torch.relu(self.conv2(h, edge_index, edge_attr))
        return self.readout(h, batch)

    def forward(self, x, edge_index, edge_attr, batch, delta):
        g = self.embed(x, edge_index, edge_attr, batch)
        g_delta = torch.cat([g, delta], dim=-1)
        psi_pred = self.psi_head(g_delta)
        force_pred = self.force_head(g_delta)
        return psi_pred, force_pred

    def set_trainable(self, encoder: bool, psi_head: bool, force_head: bool):
        """Freezes/unfreezes parameter groups by toggling requires_grad --
        used to implement the 3-stage training schedule (see train.py):
        Stage 1 trains encoder+psi_head only, Stage 2 trains force_head only
        off a frozen embedding, Stage 3 unfreezes everything for joint PINN
        finetuning."""
        for p in self.encoder.parameters():
            p.requires_grad = encoder
        for p in self.conv1.parameters():
            p.requires_grad = encoder
        for p in self.conv2.parameters():
            p.requires_grad = encoder
        for p in self.psi_head.parameters():
            p.requires_grad = psi_head
        for p in self.force_head.parameters():
            p.requires_grad = force_head
