"""
PaiNN (Polarizable Atom Interaction Neural Network) with UV prediction head.

Implements PaiNN using PyG's MessagePassing base class.
PaiNN uses equivariant message passing with scalar (s) and vector (V)
features for richer geometric representations.

Reference: Schütt et al., ICML 2021.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, radius_graph
from torch_geometric.nn.models.schnet import GaussianSmearing

from .base import MultiOutputHead, UVModelBase


class PaiNNMessage(MessagePassing):
    """PaiNN message passing layer.

    Supports Gaussian (SchNet-style) or Bessel (DimeNet-style) radial basis
    functions. Bessel basis uses learnable sinusoidal functions with a
    polynomial envelope, providing an orthogonal basis that is more
    parameter-efficient than Gaussian smearing.
    """

    def __init__(self, hidden_channels: int, num_rbf: int = 20, cutoff: float = 5.0,
                 rbf_type: str = "gaussian"):
        super().__init__(aggr="add", node_dim=0)
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.rbf_type = rbf_type

        if rbf_type == "bessel":
            from torch_geometric.nn.models.dimenet import BesselBasisLayer
            self.rbf = BesselBasisLayer(num_rbf, cutoff, envelope_exponent=5)
        else:
            self.rbf = GaussianSmearing(0.0, cutoff, num_rbf)

        # Scalar message network
        self.scalar_msg = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3 * hidden_channels),
        )

        # Filter network (radial basis -> message weights)
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3 * hidden_channels),
        )

    def forward(self, s, V, edge_index, pos):
        """
        s: scalar features [N, C]
        V: vector features [N, C, 3]
        """
        row, col = edge_index
        diff = pos[row] - pos[col]  # [E, 3]
        dist = diff.norm(dim=-1, keepdim=True)  # [E, 1]
        unit = diff / (dist + 1e-8)  # [E, 3]

        # Radial basis expansion
        rbf = self.rbf(dist.squeeze(-1))  # [E, num_rbf]
        W = self.filter_net(rbf)  # [E, 3C]

        # Cutoff: Bessel basis has polynomial envelope built-in; Gaussian needs cosine cutoff
        if self.rbf_type != "bessel":
            cutoff_val = 0.5 * (torch.cos(dist * torch.pi / self.cutoff) + 1.0)
            cutoff_val = cutoff_val * (dist < self.cutoff).float()
            W = W * cutoff_val

        # Source scalar features
        s_j = s[col]  # [E, C]
        msg = self.scalar_msg(s_j)  # [E, 3C]
        msg = msg * W  # element-wise filter

        # Split into scalar, vector gate, vector message
        ds, dV_gate, dV_msg = msg.chunk(3, dim=-1)  # each [E, C]

        # Vector update: unit direction * message
        V_j = V[col]  # [E, C, 3]
        dV = dV_msg.unsqueeze(-1) * unit.unsqueeze(1) + dV_gate.unsqueeze(-1) * V_j

        # Aggregate scalar messages
        n_nodes = s.size(0)
        s_agg = torch.zeros_like(s)
        s_agg.scatter_add_(0, row.unsqueeze(1).expand_as(ds), ds)
        s_out = s + s_agg

        # Aggregate vector messages
        V_agg = torch.zeros(n_nodes, dV.size(1), 3, device=dV.device, dtype=dV.dtype)
        V_agg.scatter_add_(0, row.unsqueeze(1).unsqueeze(2).expand_as(dV), dV)
        V_out = V + V_agg

        return s_out, V_out


class PaiNNUpdate(nn.Module):
    """PaiNN update layer (scalar-vector mixing)."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.linear_U = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.linear_V = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3 * hidden_channels),
        )

    def forward(self, s, V):
        """Mix scalar and vector features."""
        Uv = self.linear_U(V.transpose(1, 2)).transpose(1, 2)  # [N, C, 3]
        Vv = self.linear_V(V.transpose(1, 2)).transpose(1, 2)  # [N, C, 3]

        Vv_norm = Vv.norm(dim=-1)  # [N, C]
        combined = torch.cat([s, Vv_norm], dim=-1)  # [N, 2C]

        update = self.update_net(combined)  # [N, 3C]
        ds, aVV, aVs = update.chunk(3, dim=-1)

        # Scalar update
        inner = (Uv * Vv).sum(dim=-1)  # [N, C]
        s_out = s + ds + aVV * inner

        # Vector update
        V_out = V + aVs.unsqueeze(-1) * Uv

        return s_out, V_out


class PaiNNUV(UVModelBase):
    """PaiNN backbone + multi-output UV head.

    Parameters
    ----------
    hidden_channels : int
        Hidden embedding dimension. Default 128.
    num_interactions : int
        Number of message passing + update layers. Default 6.
    num_rbf : int
        Number of radial basis functions. Default 20.
    cutoff : float
        Cutoff distance (Angstroms). Default 5.0.
    head_hidden_dim : int
        Hidden dim of output MLP. Default 256.
    n_excitations : int
        Number of excitation energies to predict. Default 10.
    max_atomic_num : int
        Maximum atomic number for embedding. Default 100.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_interactions: int = 6,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        head_hidden_dim: int = 256,
        n_excitations: int = 10,
        max_atomic_num: int = 100,
        rbf_type: str = "gaussian",
    ):
        super().__init__()

        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_atomic_num, hidden_channels)

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        for _ in range(num_interactions):
            self.message_layers.append(
                PaiNNMessage(hidden_channels, num_rbf, cutoff, rbf_type=rbf_type)
            )
            self.update_layers.append(PaiNNUpdate(hidden_channels))

        self.head = MultiOutputHead(
            in_dim=hidden_channels,
            hidden_dim=head_hidden_dim,
            n_excitations=n_excitations,
        )

    def forward(self, batch) -> dict[str, torch.Tensor]:
        z, pos, edge_index = batch.z, batch.pos, batch.edge_index

        # Initial features: scalar from embedding, vector = zeros
        s = self.embedding(z)  # [N, C]
        V = torch.zeros(s.size(0), s.size(1), 3, device=s.device, dtype=s.dtype)

        for msg_layer, upd_layer in zip(self.message_layers, self.update_layers):
            s, V = msg_layer(s, V, edge_index, pos)
            s, V = upd_layer(s, V)

        # Pool scalar features to molecule level
        mol_embedding = global_mean_pool(s, batch.batch)

        return self.head(mol_embedding)
