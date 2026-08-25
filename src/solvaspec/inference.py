"""End-to-end SolvaSpec inference pipeline for a single SMILES input.

Pipeline:
    1. SMARTS preprocessor (tautomer / protonation normalisation)
    2. RDKit ETKDG conformer generation + MMFF94 optimisation (8 conformers)
    3. PaiNN-SCA forward pass (frozen backbone) on each conformer
    4. Multi-conformer spectrum averaging + NNLS decomposition back to
       50 effective (E_k, f_k) sticks
    5. For each target solvent: Spectrum Hybrid 5-fold ensemble forward,
       mean prediction + cross-fold sigma
    6. Build and return a result dict with gas-phase + per-solvent spectra
       and lambda_max +/- sigma.

Heavy dependencies (torch, chemprop, rdkit, torch_geometric) are imported
lazily so that the module doc itself can be read without them installed.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "checkpoints"
LAMBDA_GRID = np.linspace(200.0, 500.0, 301)  # nm, matches training
SIGMA_NM = 15.0
SOFT_BETA = 30.0
N_CONFORMERS = 8
N_STATES = 50
HC_EV_NM = 1239.84  # hc in eV*nm
SOLVENT_SMILES = {
    "gas": None,
    "EtOH": "CCO",
    "MeOH": "CO",
    "DMSO": "CS(C)=O",
}
# Names used by the ε-RF predictor when looking up the solvent feature.
EPS_SOLVENT_NAME = {
    "gas": "gas",
    "EtOH": "ethanol",
    "MeOH": "methanol",
    "DMSO": "dmso",
}


# ---------------------------------------------------------------------------
# Lazy singletons for the expensive model loads.
# ---------------------------------------------------------------------------

_BACKBONE = None          # PaiNN-SCA (frozen)
_HYBRID_FOLDS = None      # list of 5 HybridSpectrum instances
_DEVICE = None
_CHEMPROP_FEATURIZER = None


def _get_device():
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def _load_backbone():
    global _BACKBONE
    if _BACKBONE is not None:
        return _BACKBONE

    import torch
    from .painn_sca import PaiNNSCA

    ckpt_path = MODELS_DIR / "painn_sca_gas.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Gas-phase backbone checkpoint not found at {ckpt_path}. "
            "Place the checkpoint file before running inference."
        )

    # Instantiate with the canonical 50-excitation configuration used in the paper.
    # Checkpoint 'painn_sca_50exc_oschead_3layer/best_model.pt' was trained with a
    # 3-layer oscillator head (see CLAUDE.md), so we must reshape the head before
    # loading the state dict, otherwise nn.Linear shapes disagree.
    model = PaiNNSCA(
        hidden_channels=128,
        num_interactions=6,
        num_rbf=20,
        cutoff=5.0,
        n_excitations=N_STATES,
    )
    model.replace_osc_head("3layer")
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(_get_device())
    _BACKBONE = model
    return _BACKBONE


def _load_hybrid_folds():
    global _HYBRID_FOLDS
    if _HYBRID_FOLDS is not None:
        return _HYBRID_FOLDS

    import torch
    import torch.nn as nn
    from chemprop import nn as chemprop_nn
    from .hybrid_spectrum import HybridSpectrum

    class ChempropEncoder(nn.Module):
        def __init__(self, d_h, depth):
            super().__init__()
            self.mp = chemprop_nn.BondMessagePassing(d_h=d_h, depth=depth)
            self.agg = chemprop_nn.MeanAggregation()
            self.output_dim = self.mp.output_dim

        def forward(self, bmg):
            return self.agg(self.mp(bmg), bmg.batch)

    folds = []
    for fold in range(5):
        ckpt_path = MODELS_DIR / f"solvaspec_fold{fold}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Spectrum Hybrid fold {fold} checkpoint missing: {ckpt_path}"
            )
        solute_enc = ChempropEncoder(d_h=256, depth=3)
        solvent_enc = ChempropEncoder(d_h=128, depth=2)
        model = HybridSpectrum(
            solute_enc, solvent_enc,
            d_solute=256, d_solvent=128, n_states=N_STATES,
            sigma_nm=SIGMA_NM, soft_beta=SOFT_BETA,
        )
        state = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        model.to(_get_device())
        folds.append(model)
    _HYBRID_FOLDS = folds
    return _HYBRID_FOLDS


def _get_chemprop_featurizer():
    global _CHEMPROP_FEATURIZER
    if _CHEMPROP_FEATURIZER is None:
        from chemprop import featurizers
        _CHEMPROP_FEATURIZER = featurizers.SimpleMoleculeMolGraphFeaturizer()
    return _CHEMPROP_FEATURIZER


# ---------------------------------------------------------------------------
# Geometry / conformer generation
# ---------------------------------------------------------------------------

def _generate_conformers(smiles, n=N_CONFORMERS):
    """Return list of RDKit Mol objects, each with a single embedded conformer."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    mols = []
    for seed in range(1, n + 1):
        m = Chem.Mol(mol)  # fresh copy per seed
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useSmallRingTorsions = True
        status = AllChem.EmbedMolecule(m, params)
        if status != 0:
            continue
        try:
            AllChem.MMFFOptimizeMolecule(m, mmffVariant="MMFF94")
        except Exception:
            pass
        mols.append(m)
    if not mols:
        raise RuntimeError(f"RDKit could not embed any conformer of {smiles!r}")
    return mols


def _radius_graph_torch(pos, cutoff):
    """Pure-PyTorch radius graph for a single small molecule (no torch-cluster).

    Returns a [2, num_edges] long tensor in the torch-geometric convention,
    excluding self-loops. Suitable for ~100-atom molecules; O(n^2) but n is small.
    """
    import torch
    d = torch.cdist(pos, pos)           # [n, n] pairwise distances
    mask = (d < cutoff) & (d > 0.0)     # drop self-loops
    row, col = mask.nonzero(as_tuple=True)
    return torch.stack([row, col], dim=0)


def _mol_to_pyg(mol, cutoff=5.0):
    """Convert an RDKit Mol with one conformer to a torch_geometric Data.

    PaiNN-SCA consumes batch.edge_index (radius graph at the model cutoff);
    reproduce the same construction as the training dataset.
    """
    import torch
    from torch_geometric.data import Data

    coords = mol.GetConformer().GetPositions()
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    pos = torch.tensor(coords, dtype=torch.float32)
    batch = torch.zeros(len(z), dtype=torch.long)
    edge_index = _radius_graph_torch(pos, cutoff)
    return Data(
        z=torch.tensor(z, dtype=torch.long),
        pos=pos,
        batch=batch,
        edge_index=edge_index,
    )


# ---------------------------------------------------------------------------
# Multi-conformer spectrum averaging + NNLS back-fit
# ---------------------------------------------------------------------------

def _broaden(E_arr, f_arr, grid=LAMBDA_GRID, sigma=SIGMA_NM):
    """Sum of Gaussians at lambda = hc/E_k with heights f_k. Returns shape [len(grid)]."""
    lam_centers = HC_EV_NM / np.clip(E_arr, 0.3, 12.0)
    dev = grid[None, :] - lam_centers[:, None]
    gauss = np.exp(-0.5 * (dev / sigma) ** 2)
    return (f_arr[:, None] * gauss).sum(axis=0)


def _spectrum_level_average(per_conformer_sticks):
    """Average conformer spectra in wavelength space and back-fit 50 effective sticks.

    Returns (E_eff, f_eff) each shape [50] — compatible input for Spectrum Hybrid.
    """
    from scipy.optimize import nnls

    spectra = np.stack([
        _broaden(E, f) for (E, f) in per_conformer_sticks
    ])
    avg_spec = spectra.mean(axis=0)

    # Build reference stick grid: 50 centres uniformly spaced in energy over
    # the union of all per-conformer E ranges, and solve for their heights
    # via non-negative least squares against the averaged broadened spectrum.
    all_E = np.concatenate([E for (E, _) in per_conformer_sticks])
    E_min = np.clip(all_E.min(), 0.3, 12.0)
    E_max = np.clip(all_E.max(), E_min + 0.1, 12.0)
    E_eff = np.linspace(E_min, E_max, N_STATES)
    basis = np.stack([
        np.exp(-0.5 * ((LAMBDA_GRID - HC_EV_NM / E_k) / SIGMA_NM) ** 2)
        for E_k in E_eff
    ])  # [N_STATES, len(grid)]
    f_eff, _ = nnls(basis.T, avg_spec)
    return E_eff, f_eff, avg_spec


# ---------------------------------------------------------------------------
# Top-level predict function
# ---------------------------------------------------------------------------

@dataclass
class SolvaSpecPrediction:
    normalized_smiles: str
    rules_applied: list
    lambda_grid: np.ndarray
    # Gas-phase (PaiNN-SCA backbone, no solvent correction)
    gas_spectrum: np.ndarray              # peak-normalized broadened spectrum (a.u.)
    gas_lambda_max: float
    gas_peak_epsilon: float = float("nan")
    gas_epsilon_curve: np.ndarray | None = None
    # Solvent-aware spectra (Spectrum Hybrid head)
    etoh_spectrum: np.ndarray | None = None
    etoh_lambda_max: float = float("nan")
    etoh_sigma: float = float("nan")
    etoh_peak_epsilon: float = float("nan")
    etoh_epsilon_curve: np.ndarray | None = None
    meoh_spectrum: np.ndarray | None = None
    meoh_lambda_max: float = float("nan")
    meoh_sigma: float = float("nan")
    meoh_peak_epsilon: float = float("nan")
    meoh_epsilon_curve: np.ndarray | None = None
    dmso_spectrum: np.ndarray | None = None
    dmso_lambda_max: float = float("nan")
    dmso_sigma: float = float("nan")
    dmso_peak_epsilon: float = float("nan")
    dmso_epsilon_curve: np.ndarray | None = None
    # Backwards-compat alias — the historical "single-number" peak ε is now the
    # solvent-agnostic prediction (gas-phase null-solvent feature).
    peak_epsilon: float = float("nan")
    # Windowed ε statistics that propagate λmax uncertainty into the reported
    # amplitude. Each is {"min", "mean", "max", "lambda_lo", "lambda_hi",
    # "half_window_nm"}. Two windows per solvent: ±cross-fold σ and ±10 nm.
    etoh_eps_window_sigma: dict | None = None
    etoh_eps_window_10nm: dict | None = None
    meoh_eps_window_sigma: dict | None = None
    meoh_eps_window_10nm: dict | None = None
    dmso_eps_window_sigma: dict | None = None
    dmso_eps_window_10nm: dict | None = None
    # Integrated absorbance metrics (UV regulatory bands) in M⁻¹cm⁻¹·nm.
    # Computed in EtOH (representative solvent for sunscreens).
    integrated_epsilon_uvb: float = float("nan")  # 290–320 nm
    integrated_epsilon_uva: float = float("nan")  # 320–400 nm
    integrated_epsilon_uv: float = float("nan")   # 290–400 nm


def _soft_argmax_lambda(spectrum, beta=SOFT_BETA):
    w = np.exp(beta * spectrum / spectrum.max() + 1e-9)
    w /= w.sum()
    return float((w * LAMBDA_GRID).sum())


def _run_hybrid_ensemble(E_eff, f_eff, solvent_smiles):
    """Run 5-fold Spectrum Hybrid ensemble for a single (E_eff, f_eff, solvent)."""
    import torch
    from chemprop import data as chemprop_data
    from rdkit import Chem

    folds = _load_hybrid_folds()
    device = _get_device()
    featurizer = _get_chemprop_featurizer()

    # Placeholder solute: use the solvent-with-itself trick isn't valid; we
    # need the actual solute SMILES. The caller passes None here because
    # the solute graph is only needed for the D-MPNN encoder; E/f come from
    # the backbone. We re-construct the solute featurisation from the
    # outer predict() call so this helper is kept simple.
    raise RuntimeError(
        "_run_hybrid_ensemble should be invoked via predict(); it relies on "
        "the solute BMG constructed there."
    )


def predict(smiles: str) -> SolvaSpecPrediction:
    """End-to-end prediction for a single SMILES.

    Returns a SolvaSpecPrediction with gas-phase, EtOH, and MeOH spectra and
    lambda_max +/- cross-fold sigma for the two solvent cases.
    """
    import torch
    from chemprop import data as chemprop_data
    from rdkit import Chem

    from .smarts_preprocessor import normalize

    # 1. SMARTS preprocessing
    normalized, rules = normalize(smiles)

    # 2-3. Conformer generation + PaiNN-SCA forward
    backbone = _load_backbone()
    device = _get_device()

    conformers = _generate_conformers(normalized)
    per_conf_sticks = []
    with torch.no_grad():
        for m in conformers:
            data = _mol_to_pyg(m).to(device)
            out = backbone(data)
            # PaiNNSCA output conventions: assume keys 'excitations' and 'oscillator_strengths'
            # (or fall back to indexing — depends on checkpoint format). Adjust the line below
            # to match the checkpoint you load.
            # PaiNN-SCA forward returns dict with keys 'excitation_energies'
            # and 'oscillator_strengths' (the latter in log-space with eps=1e-3).
            # Convert log-f back to real f (clipped to >= 0).
            E = out["excitation_energies"].detach().cpu().numpy().ravel()[:N_STATES]
            log_f = out["oscillator_strengths"].detach().cpu().numpy().ravel()[:N_STATES]
            f = np.clip(np.exp(log_f) - 1e-3, 0.0, None)
            if len(E) < N_STATES:
                E = np.pad(E, (0, N_STATES - len(E)), constant_values=10.0)
                f = np.pad(f, (0, N_STATES - len(f)), constant_values=0.0)
            per_conf_sticks.append((E, f))

    # 4. Multi-conformer averaging + NNLS decomposition
    E_eff, f_eff, avg_spec = _spectrum_level_average(per_conf_sticks)
    gas_lambda_max = _soft_argmax_lambda(avg_spec)

    # 5. For each solvent, run Spectrum Hybrid 5-fold ensemble
    folds = _load_hybrid_folds()
    featurizer = _get_chemprop_featurizer()

    E_t = torch.tensor(E_eff, dtype=torch.float32, device=device).unsqueeze(0)
    f_t = torch.tensor(f_eff, dtype=torch.float32, device=device).unsqueeze(0)
    solute_mg = featurizer(Chem.MolFromSmiles(normalized))
    solute_bmg = chemprop_data.BatchMolGraph([solute_mg])
    solute_bmg.to(device)

    per_solvent = {}
    for solvent_name in ("EtOH", "MeOH", "DMSO"):
        solvent_smi = SOLVENT_SMILES[solvent_name]
        solvent_mg = featurizer(Chem.MolFromSmiles(solvent_smi))
        solvent_bmg = chemprop_data.BatchMolGraph([solvent_mg])
        solvent_bmg.to(device)

        fold_lmax = []
        fold_specs = []
        with torch.no_grad():
            for model in folds:
                out = model(E_t, f_t, solute_bmg, solvent_bmg)
                fold_lmax.append(out["lambda_max"].item())
                fold_specs.append(out["spectrum"].detach().cpu().numpy().ravel())
        fold_lmax = np.asarray(fold_lmax)
        fold_specs = np.stack(fold_specs)

        per_solvent[solvent_name] = {
            "lambda_max": float(fold_lmax.mean()),
            "sigma": float(fold_lmax.std(ddof=0)),
            "spectrum": fold_specs.mean(axis=0),
        }

    # 6. Absolute molar extinction coefficient (RF complementary model).
    #    The RF takes (solute_smiles, solvent) and predicts solvent-specific
    #    peak ε. The SolvaSpec spectra supply the shape that scales it.
    from .eps_predictor import (
        predict_peak_epsilon,
        epsilon_curve_from_normalized,
        integrated_epsilon,
        windowed_epsilon_stats,
    )

    eps_per_phase = {}                # per-solvent peak_eps (M⁻¹cm⁻¹)
    eps_curves = {}                    # per-solvent absolute ε(λ) arrays
    eps_windows_sigma = {}             # ±σ windowed-ε dicts
    eps_windows_10nm = {}              # ±10 nm windowed-ε dicts

    # Gas-phase (no solvent feature — extrapolation from liquid-phase data)
    eps_per_phase["gas"] = predict_peak_epsilon(normalized, solvent="gas")
    eps_curves["gas"] = epsilon_curve_from_normalized(avg_spec, eps_per_phase["gas"])

    for sname in ("EtOH", "MeOH", "DMSO"):
        eps_solv = predict_peak_epsilon(normalized, solvent=EPS_SOLVENT_NAME[sname])
        eps_per_phase[sname] = eps_solv
        spec = per_solvent[sname]["spectrum"]
        eps_curves[sname] = epsilon_curve_from_normalized(spec, eps_solv)
        lmax = per_solvent[sname]["lambda_max"]
        sig = per_solvent[sname]["sigma"]
        eps_windows_sigma[sname] = windowed_epsilon_stats(
            eps_curves[sname], LAMBDA_GRID, lmax, max(sig, 1.0))
        eps_windows_10nm[sname] = windowed_epsilon_stats(
            eps_curves[sname], LAMBDA_GRID, lmax, 10.0)

    # Banded integrals reported in EtOH (representative for sunscreens)
    int_uvb = integrated_epsilon(eps_curves["EtOH"], LAMBDA_GRID, 290.0, 320.0)
    int_uva = integrated_epsilon(eps_curves["EtOH"], LAMBDA_GRID, 320.0, 400.0)
    int_uv = integrated_epsilon(eps_curves["EtOH"], LAMBDA_GRID, 290.0, 400.0)

    return SolvaSpecPrediction(
        normalized_smiles=normalized,
        rules_applied=rules,
        lambda_grid=LAMBDA_GRID,
        gas_spectrum=avg_spec,
        gas_lambda_max=gas_lambda_max,
        gas_peak_epsilon=eps_per_phase["gas"],
        gas_epsilon_curve=eps_curves["gas"],
        etoh_spectrum=per_solvent["EtOH"]["spectrum"],
        etoh_lambda_max=per_solvent["EtOH"]["lambda_max"],
        etoh_sigma=per_solvent["EtOH"]["sigma"],
        etoh_peak_epsilon=eps_per_phase["EtOH"],
        etoh_epsilon_curve=eps_curves["EtOH"],
        meoh_spectrum=per_solvent["MeOH"]["spectrum"],
        meoh_lambda_max=per_solvent["MeOH"]["lambda_max"],
        meoh_sigma=per_solvent["MeOH"]["sigma"],
        meoh_peak_epsilon=eps_per_phase["MeOH"],
        meoh_epsilon_curve=eps_curves["MeOH"],
        dmso_spectrum=per_solvent["DMSO"]["spectrum"],
        dmso_lambda_max=per_solvent["DMSO"]["lambda_max"],
        dmso_sigma=per_solvent["DMSO"]["sigma"],
        dmso_peak_epsilon=eps_per_phase["DMSO"],
        dmso_epsilon_curve=eps_curves["DMSO"],
        peak_epsilon=eps_per_phase["EtOH"],   # backwards-compat alias = ε in EtOH
        etoh_eps_window_sigma=eps_windows_sigma["EtOH"],
        etoh_eps_window_10nm=eps_windows_10nm["EtOH"],
        meoh_eps_window_sigma=eps_windows_sigma["MeOH"],
        meoh_eps_window_10nm=eps_windows_10nm["MeOH"],
        dmso_eps_window_sigma=eps_windows_sigma["DMSO"],
        dmso_eps_window_10nm=eps_windows_10nm["DMSO"],
        integrated_epsilon_uvb=int_uvb,
        integrated_epsilon_uva=int_uva,
        integrated_epsilon_uv=int_uv,
    )


def predict_lambda_max_from_cached_sticks(E_gas, f_gas, solute_smiles, solvent_smiles):
    """Solvent-corrected lambda_max (nm) from pre-computed gas-phase (E, f) sticks.

    This is the exact path that produced the paper's wetlab blind-test number: the
    5-fold Spectrum Hybrid ensemble is fed the cached multi-conformer PaiNN features
    (E in eV, f linear; 50 states) together with Chemprop D-MPNN graphs of the solute
    and solvent, instead of regenerating conformers on the fly. Because the Chemprop
    encoders mean-pool over atoms, the solute/solvent graphs may be rebuilt from SMILES.
    Returns (lambda_max_mean_nm, cross_fold_sigma_nm).
    """
    import torch
    from chemprop import data as chemprop_data
    from rdkit import Chem

    folds = _load_hybrid_folds()
    device = _get_device()
    featurizer = _get_chemprop_featurizer()

    E_t = torch.as_tensor(np.asarray(E_gas, dtype=np.float32), device=device).unsqueeze(0)
    f_t = torch.as_tensor(np.asarray(f_gas, dtype=np.float32), device=device).unsqueeze(0)

    solute_bmg = chemprop_data.BatchMolGraph([featurizer(Chem.MolFromSmiles(solute_smiles))])
    solute_bmg.to(device)
    solvent_bmg = chemprop_data.BatchMolGraph([featurizer(Chem.MolFromSmiles(solvent_smiles))])
    solvent_bmg.to(device)

    lmax = []
    with torch.no_grad():
        for model in folds:
            out = model(E_t, f_t, solute_bmg, solvent_bmg)
            lmax.append(float(out["lambda_max"].item()))
    lmax = np.asarray(lmax)
    return float(lmax.mean()), float(lmax.std())
