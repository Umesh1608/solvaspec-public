"""
Base interface for all UV prediction models.

All 6 architectures implement this interface so the trainer
can handle them uniformly.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class UVModelBase(nn.Module, ABC):
    """Base class for UV absorption prediction models.

    All models take a batch of molecular graphs and return a dict
    with excitation energies, oscillator strengths, and HOMO-LUMO gap.
    """

    @abstractmethod
    def forward(self, batch) -> dict[str, torch.Tensor]:
        """Predict UV properties from a molecular graph batch.

        Parameters
        ----------
        batch : PyG Batch or framework-specific batch
            Batched molecular graphs with positions, atomic numbers, etc.

        Returns
        -------
        dict with keys:
            'excitation_energies': Tensor[B, n_exc]  (eV)
            'oscillator_strengths': Tensor[B, n_exc]
            'gap': Tensor[B, 1]  (eV)
        """

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiOutputHead(nn.Module):
    """Shared multi-output MLP head for UV predictions.

    Takes pooled molecular embeddings and produces 21 outputs:
    10 excitation energies + 10 oscillator strengths + 1 gap.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_excitations: int = 10,
        n_hidden_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_excitations = n_excitations

        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.shared_mlp = nn.Sequential(*layers)

        # Separate output heads for each target type
        self.exc_head = nn.Linear(hidden_dim, n_excitations)
        self.osc_head = nn.Linear(hidden_dim, n_excitations)
        self.gap_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor[B, in_dim]
            Pooled molecular embeddings.

        Returns
        -------
        dict with excitation_energies, oscillator_strengths, gap.
        """
        h = self.shared_mlp(x)
        return {
            "excitation_energies": self.exc_head(h),
            "oscillator_strengths": torch.relu(self.osc_head(h)),  # strengths >= 0
            "gap": self.gap_head(h),
        }
