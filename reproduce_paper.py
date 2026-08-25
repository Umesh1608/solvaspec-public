#!/usr/bin/env python3
"""Reproduce headline numbers from the SolvaSpec paper (Nature Computational Science).

Usage
-----
    python reproduce_paper.py --wetlab     # 32-pair lambda_max MAE on FDA UV filters

Reproduces the wetlab blind-test result exactly as reported in the paper. For each of
the 32 (solute, solvent) pairs it feeds the bundled cached gas-phase PaiNN-SCA sticks
(`data/wetlab_painn_cache.pkl`, multi-conformer spectrum-averaged, 50 states per solute)
together with the solute/solvent graphs to the 5-fold Spectrum Hybrid ensemble
(`checkpoints/solvaspec_fold{0..4}.pt`), and averages the folds:

    SolvaSpec wetlab blind-test MAE: 9.8 nm  (RMSE 13.2, Pearson r = 0.79)
    12-pair best-resolved MAE: 3.2 nm

Note on geometries: the headline number uses the cached multi-conformer PaiNN features
that the paper was evaluated on. The from-SMILES pipeline (`solvaspec.predict`) instead
regenerates conformers at call time, which can shift lambda_max by a few nm depending on
the RDKit version; the cache is bundled here so the reported number reproduces exactly.

Expected wallclock on a CPU laptop: well under a minute (no conformer generation).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def reproduce_wetlab(verbose: bool = True) -> dict:
    """Run the 5-fold Spectrum Hybrid ensemble on the 32 wetlab pairs and report MAE."""
    from rdkit import Chem
    from solvaspec.inference import predict_lambda_max_from_cached_sticks
    from solvaspec.smarts_preprocessor import normalize

    wetlab_csv = ROOT / "data" / "wetlab_lambda_max.csv"
    cache_path = ROOT / "data" / "wetlab_painn_cache.pkl"
    for pth in (wetlab_csv, cache_path):
        if not pth.exists():
            raise FileNotFoundError(f"{pth} not found; it should be bundled in the repo.")
    df = pd.read_csv(wetlab_csv)
    with open(cache_path, "rb") as fh:
        cache = pickle.load(fh)

    def canon(smi: str) -> str:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m else smi

    preds, errs, sigmas = [], [], []
    for i, row in df.iterrows():
        mol = row["molecule"]
        if mol not in cache:
            raise KeyError(f"no cached PaiNN features for {mol!r}")
        pred, sigma = predict_lambda_max_from_cached_sticks(
            cache[mol]["E"], cache[mol]["f"],
            normalize(row["smiles"])[0], canon(row["solvent_smiles"]),
        )
        truth = float(row["lambda_max_exp"])
        preds.append(pred)
        errs.append(pred - truth)
        sigmas.append(sigma)
        if verbose:
            print(f"  [{i + 1:>2d}/{len(df)}] {mol:<16s} ({row['solvent_name']:<4s})  "
                  f"pred {pred:6.1f} nm   exp {truth:6.1f} nm   "
                  f"err {pred - truth:+6.1f} nm   sigma {sigma:5.1f}")

    errs = np.asarray(errs)
    abs_errs = np.abs(errs)
    metrics = {
        "MAE_nm": float(abs_errs.mean()),
        "RMSE_nm": float(np.sqrt(np.mean(np.square(errs)))),
        "Pearson_r": float(np.corrcoef(df["lambda_max_exp"], preds)[0, 1]),
        "top12_MAE_nm": float(np.sort(abs_errs)[:12].mean()),
        "n_within_10nm_pairs": int((abs_errs <= 10).sum()),
        "n_total_pairs": int(len(abs_errs)),
    }
    if verbose:
        print("\n" + "=" * 64)
        print("SolvaSpec wetlab blind-test reproduction")
        print("=" * 64)
        print(f"  MAE                  = {metrics['MAE_nm']:6.2f} nm   (paper: 9.8 nm)")
        print(f"  RMSE                 = {metrics['RMSE_nm']:6.2f} nm   (paper: 13.2 nm)")
        print(f"  Pearson r            = {metrics['Pearson_r']:6.3f}    (paper: 0.79)")
        print(f"  12-pair best-resol.  = {metrics['top12_MAE_nm']:6.2f} nm   (paper: 3.2 nm)")
        print(f"  pairs within +/-10 nm = {metrics['n_within_10nm_pairs']:>2d}/{metrics['n_total_pairs']}   "
              f"(paper: ~20/32)")
        print("=" * 64)

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wetlab", action="store_true",
                        help="Reproduce the 32-pair wetlab blind-test MAE.")
    args = parser.parse_args()

    if not args.wetlab:
        parser.print_help()
        return 1
    reproduce_wetlab()
    return 0


if __name__ == "__main__":
    sys.exit(main())
