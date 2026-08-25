# Model checkpoints

All checkpoints needed to reproduce the headline wetlab result are **bundled in
this directory** — no download step is required.

| File | Role | Size |
|---|---|---|
| `painn_sca_gas.pt` | PaiNN-SCA gas-phase backbone (50 states, 3-layer oscillator head) | 6.1 MB |
| `solvaspec_fold0.pt` … `solvaspec_fold4.pt` | Spectrum Hybrid 5-fold ensemble (solvent correction) | 1.2 MB each |
| `eps_rf_solvent.pkl` | Solvent-aware ε Random Forest (absolute molar extinction) | 46 MB |

Together these drive `predict()` (`src/solvaspec/inference.py`) and the
`python reproduce_paper.py --wetlab` headline reproduction.

## Sanity check

```bash
python -c "from solvaspec import predict; print(round(predict('CCO').gas_lambda_max, 1))"
```

Loads every checkpoint and runs a single-molecule inference; prints a number if
the pipeline is wired up correctly.
