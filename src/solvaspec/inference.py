"""End-to-end SolvaSpec inference: SMILES -> calibrated solvent-aware spectrum.

The actual model implementation (PaiNN-SCA + Spectrum Hybrid + ε Random Forest)
lives in the parent monorepo `uv_predict` package; this module wires the
pieces together for the public release.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------- API


@dataclasses.dataclass
class SpectrumResult:
    """Single-molecule prediction returned by ``predict_solvent_spectrum``."""

    smiles: str
    solvent: str
    wavelength_nm: np.ndarray              # shape (N,)
    absorbance: np.ndarray                 # shape (N,) in M^-1 cm^-1
    lambda_max_nm: float
    eps_max: float                         # M^-1 cm^-1
    uva: float                             # integrated absorbance over 320-400 nm
    uvb: float                             # integrated absorbance over 280-320 nm
    cross_fold_sigma_nm: Optional[float]   # 5-fold ensemble lambda_max sigma

    def plot(self, path: str | Path = "predicted_spectrum.png") -> Path:
        """Save a quick PNG of the predicted curve."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(self.wavelength_nm, self.absorbance, color="#2E86AB", lw=1.6)
        ax.axvline(self.lambda_max_nm, color="#cc4444", ls="--", lw=0.9,
                   label=fr"$\lambda_\mathrm{{max}}$ = {self.lambda_max_nm:.0f} nm")
        ax.set_xlabel(r"$\lambda$ (nm)")
        ax.set_ylabel(r"absorbance ($\mathrm{M}^{-1}\mathrm{cm}^{-1}$)")
        ax.set_title(f"{self.smiles}  in  {self.solvent}")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.3)
        path = Path(path)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path


def predict_solvent_spectrum(
    smiles: str,
    solvent: str = "ethanol",
    *,
    n_conformers: int = 8,
    return_uncertainty: bool = True,
) -> SpectrumResult:
    """Predict a solvent-aware UV-Vis spectrum for one molecule.

    Parameters
    ----------
    smiles : str
        Solute SMILES.
    solvent : str, default ``"ethanol"``
        Solvent identifier. Accepts SMILES (``"CCO"``) or one of the
        named shortcuts ``"ethanol" | "methanol" | "water" | "DMSO" | "gas"``.
        ``"gas"`` skips the Spectrum Hybrid head and returns the calibrated
        gas-phase backbone spectrum.
    n_conformers : int, default 8
        Number of ETKDG conformers to average over.
    return_uncertainty : bool, default True
        If ``True``, run the 5-fold ensemble and report the cross-fold
        standard deviation of λ_max as an out-of-distribution signal.

    Returns
    -------
    SpectrumResult
    """
    # -- IMPORTANT --
    # The full implementation lives in the project's main repo, in
    # `src/uv_predict/`. The release version of SolvaSpec calls into that
    # package via a thin wrapper. To run, the main repo must be on PYTHONPATH:
    #     export PYTHONPATH=/path/to/solvaspec/main_repo/src
    # OR the user can install the repo as a package:
    #     pip install -e <path-to-uv_predict>
    #
    # For the public release, the relevant model files are:
    #     src/uv_predict/models/painn_sca.py        # PaiNN-SCA backbone
    #     src/uv_predict/models/painn_uv.py         # PaiNN message/update layers
    #     src/uv_predict/evaluation/spectrum.py     # broadening utilities
    # plus the Spectrum Hybrid head and ε RF wrappers below.

    from .model import load_full_pipeline
    from .pipeline import run_pipeline

    pipeline = load_full_pipeline()
    return run_pipeline(
        pipeline,
        smiles=smiles,
        solvent=solvent,
        n_conformers=n_conformers,
        return_uncertainty=return_uncertainty,
    )
