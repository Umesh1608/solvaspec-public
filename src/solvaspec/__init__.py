"""SolvaSpec: deep learning for solvent-aware UV-Vis spectra and absolute ε_max.

Public entry point:
    >>> from solvaspec import predict
    >>> result = predict("CC(C)(C)c1ccc(cc1)/C(O)=C\\\\C(=O)c1ccc(OC)cc1")
    >>> result.meoh_lambda_max  # ≈ 358 nm for avobenzone enol in MeOH
"""
from .inference import predict, SolvaSpecPrediction

__version__ = "1.0.0"
__all__ = ["predict", "SolvaSpecPrediction"]
