"""Model loaders: PaiNN-SCA backbone, Spectrum Hybrid head, ε Random Forest."""
from __future__ import annotations

from pathlib import Path

import torch
import joblib

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

C_GAS_AMP = 0.25      # gas-phase oscillator-strength calibration
C_HYBRID_AMP = 0.040  # solvent-aware (Hybrid) oscillator-strength calibration


def load_painn_sca(
    ckpt: Path = CHECKPOINT_DIR / "painn_sca_50exc_oschead_3layer.pt",
    device: str = "cpu",
):
    """Load the PaiNN-SCA backbone with the 3-layer oscillator-strength head."""
    try:
        from uv_predict.models import build_model
    except ImportError as e:
        raise ImportError(
            "The uv_predict package was not found. "
            "Either install the main monorepo (`pip install -e <path-to-uv_predict>`) "
            "or ensure src/ of this repo is on PYTHONPATH."
        ) from e

    model = build_model(
        "painn_sca",
        hidden_channels=128,
        num_interactions=6,
        num_rbf=20,
        cutoff=5.0,
        n_excitations=50,
    )
    model.replace_osc_head(head_type="3layer")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model.to(device)


def load_spectrum_hybrid(
    ckpt: Path = CHECKPOINT_DIR / "spectrum_hybrid_5fold.pt",
    device: str = "cpu",
):
    """Load the Spectrum Hybrid head ensemble (5 folds)."""
    state = torch.load(ckpt, map_location=device, weights_only=False)
    return state  # dict of fold_id -> SpectrumHybridHead


def load_eps_rf(ckpt: Path = CHECKPOINT_DIR / "eps_random_forest.joblib"):
    """Load the absolute ε_max Random Forest."""
    return joblib.load(ckpt)


def load_full_pipeline(device: str = "cpu") -> dict:
    """Load all three model components into a dict."""
    return {
        "backbone": load_painn_sca(device=device),
        "hybrid": load_spectrum_hybrid(device=device),
        "eps_rf": load_eps_rf(),
        "c_gas_amp": C_GAS_AMP,
        "c_hybrid_amp": C_HYBRID_AMP,
    }
