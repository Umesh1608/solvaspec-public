"""Sanity checks: load every checkpoint, run a single-molecule inference,
and print PASS / FAIL.  Useful right after downloading the artefacts.
"""
from __future__ import annotations

import sys

TEST_SMILES = "CC(C)(C)c1ccc(cc1)/C(O)=C\\C(=O)c1ccc(OC)cc1"  # avobenzone enol


def verify_checkpoints() -> bool:
    """Run a one-molecule end-to-end pass; return True if everything works."""
    try:
        from .inference import predict_solvent_spectrum
        result = predict_solvent_spectrum(
            smiles=TEST_SMILES,
            solvent="methanol",
            n_conformers=2,         # fast smoke-test
            return_uncertainty=False,
        )
        ok = (
            300 <= result.lambda_max_nm <= 400
            and result.eps_max > 0
            and len(result.wavelength_nm) > 0
        )
        if ok:
            print(f"PASS  avobenzone (enol, MeOH): "
                  f"λ_max = {result.lambda_max_nm:.1f} nm, "
                  f"ε_max = {result.eps_max:.2e} M^-1 cm^-1")
        else:
            print(f"FAIL  unexpected output: {result}")
        return ok
    except Exception as e:
        print(f"FAIL  {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if verify_checkpoints() else 1)
