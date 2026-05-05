"""
PaiNN + State Cross-Attention (PaiNN-SCA) for UV prediction.

Novel architecture that combines:
1. PaiNN equivariant backbone for local atomic interactions
2. Learnable state query tokens (S1-S10) that cross-attend to
   atom embeddings to predict per-state excited properties.

Each state token learns which atoms and interactions contribute
to that specific electronic excitation — physically motivated by
the fact that different excited states involve different atomic
orbital contributions (pi->pi*, n->pi*, charge-transfer, etc.).

This is the first architecture to use state-query cross-attention
for excited-state property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .base import UVModelBase
from .painn_uv import PaiNNMessage, PaiNNUpdate


class StateCrossAttention(nn.Module):
    """Cross-attention from learnable state query tokens to atom features.

    Each state token (representing an electronic excitation S1...S_n)
    attends to the atomic scalar features to gather state-specific
    molecular information.

    Parameters
    ----------
    hidden_channels : int
        Dimension of atom embeddings and state tokens.
    n_states : int
        Number of excited states to predict. Default 50.
    n_heads : int
        Number of attention heads. Default 4.
    dropout : float
        Attention dropout rate. Default 0.1.
    """

    def __init__(
        self,
        hidden_channels: int,
        n_states: int = 50,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_states = n_states
        self.n_heads = n_heads
        self.head_dim = hidden_channels // n_heads
        assert hidden_channels % n_heads == 0

        # Learnable state query tokens — each represents one electronic state
        self.state_queries = nn.Parameter(
            torch.randn(n_states, hidden_channels) * 0.02
        )

        # Projections for multi-head cross-attention
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)

        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(hidden_channels)
        self.norm_k = nn.LayerNorm(hidden_channels)

        # Per-state output heads: each state token predicts (energy, strength)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.strength_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross-attend state tokens to atom features.

        Parameters
        ----------
        atom_features : Tensor[N_atoms, C]
            Per-atom scalar features from PaiNN backbone.
        batch_idx : Tensor[N_atoms]
            Batch assignment for each atom.

        Returns
        -------
        excitation_energies : Tensor[B, n_states]
        oscillator_strengths : Tensor[B, n_states]
        attention_weights : Tensor[B, n_states, max_atoms]
            For interpretability — which atoms drive each state.
        """
        n_molecules = batch_idx.max().item() + 1
        device = atom_features.device

        # Expand state queries for each molecule in the batch
        # Q: [B, n_states, C]
        Q = self.state_queries.unsqueeze(0).expand(n_molecules, -1, -1)
        Q = self.norm_q(Q)
        Q = self.q_proj(Q)

        # Prepare per-molecule atom features (padded)
        # Find max atoms per molecule for padding
        counts = torch.bincount(batch_idx, minlength=n_molecules)
        max_atoms = counts.max().item()

        # Pad atom features into [B, max_atoms, C]
        K_padded = torch.zeros(n_molecules, max_atoms, atom_features.size(-1), device=device)
        mask = torch.ones(n_molecules, max_atoms, dtype=torch.bool, device=device)  # True = masked

        offset = 0
        for mol_idx in range(n_molecules):
            n = counts[mol_idx].item()
            K_padded[mol_idx, :n] = atom_features[offset:offset + n]
            mask[mol_idx, :n] = False
            offset += n

        K_padded = self.norm_k(K_padded)
        K = self.k_proj(K_padded)  # [B, max_atoms, C]
        V = self.v_proj(K_padded)  # [B, max_atoms, C]

        # Multi-head cross-attention
        B, S, C = Q.shape
        _, A, _ = K.shape

        Q = Q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, S, d]
        K = K.view(B, A, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, A, d]
        V = V.view(B, A, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, A, d]

        # Attention scores: [B, H, S, A]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask (padded atoms get -inf)
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, A]
        scores = scores.masked_fill(mask_expanded, float("-inf"))

        # Numerically stable softmax: clamp + subtract max
        scores = scores.clamp(-50.0, 50.0)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0)
        attn_weights = self.dropout(attn_weights)

        # Aggregate: [B, H, S, d]
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, S, C)
        context = self.out_proj(context)  # [B, n_states, C]

        # Per-state predictions
        exc_energies = self.energy_head(context).squeeze(-1)  # [B, n_states]
        osc_strengths = self.strength_head(context).squeeze(-1)  # [B, n_states] log-space

        # Average attention over heads for interpretability: [B, n_states, max_atoms]
        attn_map = attn_weights.mean(dim=1)

        return exc_energies, osc_strengths, attn_map


class PaiNNSCA(UVModelBase):
    """PaiNN backbone + State Cross-Attention for UV prediction.

    Novel architecture: equivariant local features (PaiNN) + learnable
    state query tokens that cross-attend to discover per-state atomic
    contributions.

    Parameters
    ----------
    hidden_channels : int
        Hidden embedding dimension. Default 128.
    num_interactions : int
        Number of PaiNN message passing layers. Default 6.
    num_rbf : int
        Number of radial basis functions. Default 20.
    cutoff : float
        Cutoff distance (Angstroms). Default 5.0.
    n_excitations : int
        Number of excited states to predict. Default 50.
    n_heads : int
        Number of attention heads. Default 4.
    max_atomic_num : int
        Maximum atomic number for embedding. Default 100.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_interactions: int = 6,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        n_excitations: int = 50,
        n_heads: int = 4,
        max_atomic_num: int = 100,
        rbf_type: str = "gaussian",
    ):
        super().__init__()

        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_atomic_num, hidden_channels)

        # PaiNN backbone (equivariant message passing)
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        for _ in range(num_interactions):
            self.message_layers.append(PaiNNMessage(hidden_channels, num_rbf, cutoff, rbf_type=rbf_type))
            self.update_layers.append(PaiNNUpdate(hidden_channels))

        # Novel: State Cross-Attention layer
        self.state_attention = StateCrossAttention(
            hidden_channels=hidden_channels,
            n_states=n_excitations,
            n_heads=n_heads,
        )

        # Separate gap prediction from pooled features
        self.gap_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, batch) -> dict[str, torch.Tensor]:
        z, pos, edge_index = batch.z, batch.pos, batch.edge_index

        # PaiNN backbone
        s = self.embedding(z)
        V = torch.zeros(s.size(0), s.size(1), 3, device=s.device, dtype=s.dtype)

        for msg_layer, upd_layer in zip(self.message_layers, self.update_layers):
            s, V = msg_layer(s, V, edge_index, pos)
            s, V = upd_layer(s, V)

        # State Cross-Attention: each state token attends to atom features
        exc_energies, osc_strengths, attn_map = self.state_attention(s, batch.batch)

        # HOMO-LUMO gap from global pooling
        mol_embedding = global_mean_pool(s, batch.batch)
        gap = self.gap_head(mol_embedding)

        result = {
            "excitation_energies": exc_energies,
            "oscillator_strengths": osc_strengths,
            "gap": gap,
        }

        # Store attention maps for interpretability (optional)
        if not self.training:
            result["state_attention_maps"] = attn_map

        return result

    def freeze_except_osc(self):
        """Freeze all parameters except the oscillator strength head."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.state_attention.strength_head.parameters():
            param.requires_grad = True

    def replace_osc_head(self, head_type="2layer"):
        """Replace the oscillator head with a new architecture."""
        C = self.state_attention.strength_head[0].in_features  # hidden_channels
        if head_type == "2layer":
            self.state_attention.strength_head = nn.Sequential(
                nn.Linear(C, C),
                nn.SiLU(),
                nn.Linear(C, 1),
            )
        elif head_type == "3layer":
            self.state_attention.strength_head = nn.Sequential(
                nn.Linear(C, C),
                nn.SiLU(),
                nn.Linear(C, C // 2),
                nn.SiLU(),
                nn.Linear(C // 2, 1),
            )
