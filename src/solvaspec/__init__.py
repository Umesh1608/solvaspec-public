"""SolvaSpec: deep learning for solvent-aware UV-Vis spectra and ε_max."""
from .inference import predict_solvent_spectrum, SpectrumResult
from .model import load_painn_sca, load_spectrum_hybrid, load_eps_rf
from .verify import verify_checkpoints

__version__ = "1.0.0"
__all__ = [
    "predict_solvent_spectrum",
    "SpectrumResult",
    "load_painn_sca",
    "load_spectrum_hybrid",
    "load_eps_rf",
    "verify_checkpoints",
]
