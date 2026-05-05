#!/usr/bin/env python3
"""Reproduce headline numbers from the SolvaSpec paper (npj Comp Mat).

Usage
-----
    python reproduce_paper.py --wetlab     # 32-pair λmax MAE on FDA UV filters

Reads the PaiNN-SCA, Spectrum Hybrid, and ε Random Forest checkpoints from
``checkpoints/`` and the wetlab CSV from ``data/wetlab_lambda_max.csv``,
runs the full pipeline on every (solute, solvent) pair, and prints the
paper's headline metrics:

    SolvaSpec wetlab blind-test MAE: 9.8 nm  (RMSE 13.2, Pearson r = 0.79)
    12-pair best-resolved MAE: 3.2 nm

Expected wallclock on a CPU laptop: ~5 min for all 32 pairs (8 conformers ×
5-fold ensemble × 32 pairs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def reproduce_wetlab(verbose: bool = True) -> dict:
    """Run SolvaSpec on the 32 in-house wetlab measurements and report MAE."""
    from solvaspec import predict

    wetlab_csv = ROOT / "data" / "wetlab_lambda_max.csv"
    if not wetlab_csv.exists():
        raise FileNotFoundError(
            f"{wetlab_csv} not found. "
            "Download the wetlab archive from the Zenodo DOI listed in "
            "data/README.md, or copy it from the development repository."
        )
    df = pd.read_csv(wetlab_csv)

    preds, errs, sigmas = [], [], []
    for i, row in df.iterrows():
        smi = row["smiles"]
        solv = row["solvent_name"]   # "EtOH" or "MeOH"
        truth = float(row["lambda_max_exp"])
        if verbose:
            print(f"  [{i + 1:>2d}/{len(df)}] {row['molecule']:<16s} ({solv})  ...", end="", flush=True)
        result = predict(smi)
        if solv == "EtOH":
            pred = result.etoh_lambda_max
            sigma = result.etoh_sigma
        elif solv == "MeOH":
            pred = result.meoh_lambda_max
            sigma = result.meoh_sigma
        else:
            raise ValueError(f"unsupported solvent {solv}")
        preds.append(pred)
        errs.append(pred - truth)
        sigmas.append(sigma)
        if verbose:
            print(f"  pred {pred:6.1f} nm   exp {truth:6.1f} nm   "
                  f"err {pred - truth:+6.1f} nm   σ {sigma:5.1f}")

    df["lambda_max_pred_nm"] = preds
    df["error_nm"] = errs
    df["cross_fold_sigma_nm"] = sigmas

    abs_errs = np.abs(errs)
    abs_errs_sorted = np.sort(abs_errs)
    top12 = abs_errs_sorted[:12]

    metrics = {
        "MAE_nm": float(abs_errs.mean()),
        "RMSE_nm": float(np.sqrt(np.mean(np.square(errs)))),
        "Pearson_r": float(np.corrcoef(df["lambda_max_exp"], preds)[0, 1]),
        "top12_MAE_nm": float(top12.mean()),
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
        print(f"  pairs within ±10 nm  = {metrics['n_within_10nm_pairs']:>2d}/{metrics['n_total_pairs']}   "
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
    if args.wetlab:
        reproduce_wetlab()
    return 0


if __name__ == "__main__":
    sys.exit(main())
