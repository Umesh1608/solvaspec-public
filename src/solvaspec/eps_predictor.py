"""Molar extinction coefficient (ε) predictor — solvent-aware version.

A complementary model to the SolvaSpec PaiNN-SCA + Spectrum Hybrid pipeline.
Whereas SolvaSpec excels at predicting spectral shape and peak position, the
per-state oscillator strengths f_k from the log-space training objective are
near-uniform across molecules and therefore cannot resolve absolute amplitude
on a per-compound basis.

This module supplies that missing axis. A Random Forest takes:
- Solute Morgan-2 fingerprint (1024 bits)
- 15 RDKit descriptors of the solute
- Solvent Morgan-2 fingerprint (256 bits)
- 6 solvent descriptors

and predicts log10(ε_max) at the absorption peak.

Calibration set: 6,283 (molecule, solvent) pairs from PhotochemCAD and Joung
et al., 80/10/10 random split.

Held-out test performance:
    Overall: factor 1.44× error  |  Spearman ρ = 0.84  |  R² = 0.72
    By solvent:  MeOH 1.28×  acetone 1.15×  ACN 1.42×  THF 1.38×
                 EtOH 1.53×  DCM 1.58×      DMSO 1.58×

Gas-phase ε is approximated by passing an all-zero solvent vector — there is
no experimental gas-phase ε at scale to train on, so this is an extrapolation
from liquid-phase measurements.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RF_PATH = MODELS_DIR / "eps_rf_solvent.pkl"

# Solvent name → SMILES mapping. Recognised solvents are featurised via Morgan
# fingerprint of the SMILES; unrecognised names fall back to "use the SMILES
# as if it were already SMILES", which lets the user pass any RDKit-parseable
# solvent SMILES directly.
NAME_TO_SMILES = {
    "methanol": "CO",         "meoh": "CO",         "ch3oh": "CO",
    "ethanol": "CCO",         "etoh": "CCO",
    "water": "O",             "h2o": "O",            "aqueous": "O",   "buffer": "O",   "pbs": "O",
    "cyclohexane": "C1CCCCC1",
    "toluene": "Cc1ccccc1",
    "acetonitrile": "CC#N",   "acn": "CC#N",         "ch3cn": "CC#N",
    "thf": "C1CCOC1",         "tetrahydrofuran": "C1CCOC1",
    "dichloromethane": "ClCCl", "dcm": "ClCCl",      "ch2cl2": "ClCCl",
    "chloroform": "ClC(Cl)Cl", "chcl3": "ClC(Cl)Cl",
    "dmso": "CS(C)=O",        "dimethyl_sulfoxide": "CS(C)=O",
    "dmf": "CN(C)C=O",        "dimethylformamide": "CN(C)C=O",
    "diethyl_ether": "CCOCC", "et2o": "CCOCC",       "ether": "CCOCC",
    "isopropanol": "CC(C)O",  "ipa": "CC(C)O",       "ipoh": "CC(C)O",
    "acetone": "CC(=O)C",
    "1-propanol": "CCCO",     "propanol": "CCCO",
    "n-butanol": "CCCCO",     "butanol": "CCCCO",
    "tert-butanol": "CC(C)(C)O",
    "ethyl_acetate": "CCOC(=O)C", "etoac": "CCOC(=O)C",
    "carbon_tetrachloride": "ClC(Cl)(Cl)Cl", "ccl4": "ClC(Cl)(Cl)Cl",
    "1,4-dioxane": "C1CCOCC1O", "dioxane": "C1CCOCC1O",
    "benzene": "c1ccccc1",
    "n-hexane": "CCCCCC",     "hexane": "CCCCCC",
    "n-heptane": "CCCCCCC",   "heptane": "CCCCCCC",
    "pyridine": "c1ccncc1",
    "trifluoroethanol": "OCC(F)(F)F", "tfe": "OCC(F)(F)F",
    "ethylene_glycol": "OCCO",
}

SOLVENT_BIT_DIM = 256
SOLVENT_DESC_DIM = 6
SOLVENT_DIM = SOLVENT_BIT_DIM + SOLVENT_DESC_DIM  # 262

_RF = None


def _load_rf():
    global _RF
    if _RF is None:
        with open(RF_PATH, "rb") as f:
            _RF = pickle.load(f)
    return _RF


def featurize_solute(smiles: str) -> np.ndarray | None:
    """1024-bit Morgan FP + 15 RDKit descriptors. Returns None on parse error."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
    fp_arr = np.zeros(1024, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    desc = np.array([
        Descriptors.MolWt(m), Descriptors.MolLogP(m),
        Descriptors.NumAromaticRings(m), Descriptors.NumAliphaticRings(m),
        Descriptors.NumRotatableBonds(m), Descriptors.NumHDonors(m),
        Descriptors.NumHAcceptors(m), Descriptors.TPSA(m),
        Descriptors.NumHeteroatoms(m), Descriptors.HeavyAtomCount(m),
        Descriptors.FractionCSP3(m), Descriptors.NumAliphaticHeterocycles(m),
        Descriptors.NumAromaticHeterocycles(m),
        Descriptors.RingCount(m), Descriptors.NumSaturatedRings(m),
    ], dtype=np.float32)
    return np.concatenate([fp_arr, desc])


def featurize_solvent(name_or_smiles: str | None) -> np.ndarray:
    """Return a 262-dim solvent feature vector. All zeros for gas / unknown."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors

    if name_or_smiles is None or name_or_smiles == "" or str(name_or_smiles).lower() == "gas":
        return np.zeros(SOLVENT_DIM, dtype=np.float32)
    s = str(name_or_smiles).strip().lower()
    smi = NAME_TO_SMILES.get(s, name_or_smiles)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(SOLVENT_DIM, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=SOLVENT_BIT_DIM)
    fp_arr = np.zeros(SOLVENT_BIT_DIM, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    desc = np.array([
        Descriptors.MolWt(m), Descriptors.MolLogP(m),
        Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
        Descriptors.TPSA(m), Descriptors.NumAromaticRings(m),
    ], dtype=np.float32)
    return np.concatenate([fp_arr, desc])


def predict_peak_epsilon(solute_smiles: str, solvent: str | None = None) -> float:
    """Predict ε_max in M⁻¹cm⁻¹ for a given solute and solvent.

    Parameters
    ----------
    solute_smiles : str
        Solute SMILES.
    solvent : str | None
        Solvent name (e.g. "ethanol", "dmso") or SMILES (e.g. "CCO"), or
        "gas" / None for gas-phase approximation (extrapolation).
    """
    solute_feat = featurize_solute(solute_smiles)
    if solute_feat is None:
        return float("nan")
    solvent_feat = featurize_solvent(solvent)
    X = np.concatenate([solute_feat, solvent_feat]).reshape(1, -1)
    rf = _load_rf()
    log10_eps = float(rf.predict(X)[0])
    return float(10.0 ** log10_eps)


def epsilon_curve_from_normalized(
    normalized_spectrum: np.ndarray, peak_eps: float
) -> np.ndarray:
    """Convert a peak-normalized spectrum to an absolute ε(λ) curve.

    By construction max(returned curve) == peak_eps.
    """
    if peak_eps != peak_eps:  # NaN
        return np.full_like(normalized_spectrum, np.nan, dtype=np.float64)
    s = normalized_spectrum.astype(np.float64)
    s_max = float(s.max())
    if s_max <= 0.0:
        return np.zeros_like(s)
    return s / s_max * peak_eps


def integrated_epsilon(
    eps_curve: np.ndarray, lambda_grid: np.ndarray,
    lambda_lo_nm: float, lambda_hi_nm: float,
) -> float:
    """Integrate ε(λ) over [lambda_lo, lambda_hi] in M⁻¹cm⁻¹·nm.

    Useful for UVA (320–400 nm) and UVB (290–320 nm) integrated absorbance
    metrics in UV filter screening.
    """
    mask = (lambda_grid >= lambda_lo_nm) & (lambda_grid <= lambda_hi_nm)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(eps_curve[mask], lambda_grid[mask]))


def windowed_epsilon_stats(
    eps_curve: np.ndarray, lambda_grid: np.ndarray,
    lambda_center_nm: float, half_window_nm: float,
) -> dict:
    """Return min / mean / max of ε(λ) within [center ± half_window]."""
    nan = float("nan")
    if eps_curve is None or len(eps_curve) == 0:
        return {"min": nan, "mean": nan, "max": nan,
                "lambda_lo": nan, "lambda_hi": nan, "half_window_nm": float(half_window_nm)}
    lo = lambda_center_nm - half_window_nm
    hi = lambda_center_nm + half_window_nm
    mask = (lambda_grid >= lo) & (lambda_grid <= hi)
    if mask.sum() == 0:
        return {"min": nan, "mean": nan, "max": nan,
                "lambda_lo": float(lo), "lambda_hi": float(hi),
                "half_window_nm": float(half_window_nm)}
    seg = eps_curve[mask]
    return {
        "min": float(seg.min()),
        "mean": float(seg.mean()),
        "max": float(seg.max()),
        "lambda_lo": float(lo),
        "lambda_hi": float(hi),
        "half_window_nm": float(half_window_nm),
    }
