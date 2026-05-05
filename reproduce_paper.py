#!/usr/bin/env python3
"""Reproduce headline numbers from the SolvaSpec paper (npj Comp Mat).

Usage
-----

    python reproduce_paper.py --wetlab          # 32-pair blind-test MAE
    python reproduce_paper.py --gdb9ex          # GDB-9-Ex SRMSE
    python reproduce_paper.py --all             # everything

The script reads checkpoints from `checkpoints/` and data from `data/`,
runs the full SolvaSpec pipeline (PaiNN-SCA backbone + Spectrum Hybrid
head + ε Random Forest) on the held-out evaluation set, and prints the
paper's headline metrics.

Expected wallclock on a CPU laptop:
  --wetlab    : ~30 seconds
  --gdb9ex    : ~10 minutes (96,586 molecules)
  --all       : ~10 minutes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the local solvaspec package is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


# ----------------------------------------------------------- evaluation tasks


def reproduce_wetlab(verbose: bool = True) -> dict:
    """Run SolvaSpec on the 32 in-house wetlab measurements and report MAE.

    Reproduces the headline number from the paper:
        wetlab_lambda_max_MAE = 9.8 nm
        12-pair best-resolved MAE = 3.2 nm
    """
    from solvaspec import predict_solvent_spectrum

    wetlab_csv = ROOT / "data" / "wetlab_lambda_max.csv"
    if not wetlab_csv.exists():
        raise FileNotFoundError(
            f"{wetlab_csv} not found. "
            "Download the wetlab archive from the Zenodo DOI listed in "
            "data/README.md and place it in this directory."
        )
    df = pd.read_csv(wetlab_csv)

    preds, errs = [], []
    for _, row in df.iterrows():
        smi = row["smiles"]
        solv = row["solvent"]   # "ethanol" or "methanol"
        truth = row["lambda_max_exp_nm"]
        pred = predict_solvent_spectrum(
            smi, solv,
            n_conformers=8,
            return_uncertainty=True,
        )
        preds.append(pred.lambda_max_nm)
        errs.append(pred.lambda_max_nm - truth)

    df["lambda_max_pred_nm"] = preds
    df["error_nm"] = errs
    abs_errs = np.abs(errs)
    abs_errs_sorted = np.sort(abs_errs)
    top12 = abs_errs_sorted[:12]

    metrics = {
        "MAE_nm": float(abs_errs.mean()),
        "RMSE_nm": float(np.sqrt(np.mean(np.square(errs)))),
        "Pearson_r": float(np.corrcoef(df["lambda_max_exp_nm"], preds)[0, 1]),
        "top12_MAE_nm": float(top12.mean()),
        "n_within_10nm": int((abs_errs <= 10).sum()),
        "n_total": int(len(abs_errs)),
    }
    if verbose:
        print("=" * 60)
        print("SolvaSpec wetlab blind-test reproduction")
        print("=" * 60)
        print(f"  MAE              = {metrics['MAE_nm']:.2f} nm  (paper: 9.8 nm)")
        print(f"  RMSE             = {metrics['RMSE_nm']:.2f} nm  (paper: 13.2 nm)")
        print(f"  Pearson r        = {metrics['Pearson_r']:.3f}  (paper: 0.79)")
        print(f"  12-pair best MAE = {metrics['top12_MAE_nm']:.2f} nm  (paper: 3.2 nm)")
        print(f"  within ±10 nm    = {metrics['n_within_10nm']}/{metrics['n_total']}  (paper: 10/16 mols, ~20/32 pairs)")
    return metrics


def reproduce_gdb9ex(verbose: bool = True) -> dict:
    """Run PaiNN-SCA on the 96,586-molecule GDB-9-Ex held-out test and
    reproduce the headline SRMSE number (mean 0.0047, median 0.0023)."""
    from solvaspec.gdb9ex_eval import evaluate_gdb9ex
    metrics = evaluate_gdb9ex(verbose=verbose)
    if verbose:
        print("=" * 60)
        print("PaiNN-SCA GDB-9-Ex reproduction")
        print("=" * 60)
        print(f"  mean SRMSE   = {metrics['srmse_mean']:.4f}  (paper: 0.0047)")
        print(f"  median SRMSE = {metrics['srmse_median']:.4f}  (paper: 0.0023)")
        print(f"  n_molecules  = {metrics['n']}  (paper: 96,586)")
    return metrics


# --------------------------------------------------------------- entry point


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wetlab", action="store_true",
                        help="Reproduce the 32-pair wetlab blind-test MAE.")
    parser.add_argument("--gdb9ex", action="store_true",
                        help="Reproduce GDB-9-Ex SRMSE on 96,586 molecules.")
    parser.add_argument("--all", action="store_true", help="Run every reproducer.")
    args = parser.parse_args()

    if not (args.wetlab or args.gdb9ex or args.all):
        parser.print_help()
        return 1

    if args.wetlab or args.all:
        reproduce_wetlab()
    if args.gdb9ex or args.all:
        reproduce_gdb9ex()
    return 0


if __name__ == "__main__":
    sys.exit(main())
