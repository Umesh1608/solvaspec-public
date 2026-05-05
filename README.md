# SolvaSpec

> An experimentally validated deep-learning prediction platform for solvent-aware UV-Vis spectra and molar extinction coefficients.

This repository accompanies the paper *SolvaSpec: An experimentally validated deep-learning prediction platform for solvent-aware UV-Vis spectra and extinction coefficients* (npj Computational Materials, in submission).

For a one-click interactive demo, use the public HuggingFace Space:
**https://huggingface.co/spaces/MWBC/SolvaSpec_MWBC**

## What this repo contains

| Path | Contents |
|---|---|
| `src/solvaspec/` | Core PaiNN-SCA + Spectrum Hybrid head + ε Random Forest pipeline |
| `checkpoints/` | Trained model artefacts (download instructions inside) |
| `data/` | F13 splits, wetlab measurements, ε calibration set indices |
| `notebooks/` | Worked-example notebooks (predict a molecule; reproduce wetlab results) |
| `reproduce_paper.py` | **Single command that regenerates the headline wetlab MAE number** |
| `scripts/` | All figure-generation and analysis scripts used in the paper |
| `figures/` | Publication-quality PDFs of all main + supplementary figures |

## 30-second quickstart

```bash
git clone https://github.com/Umesh1608/solvaspec-public
cd solvaspec
conda env create -f environment.yml
conda activate solvaspec
python reproduce_paper.py --wetlab
```

Expected output:
```
SolvaSpec wetlab blind-test MAE: 9.8 nm  (RMSE 13.2, Pearson r = 0.79)
12-pair best-resolved MAE: 3.2 nm
```

## Predicting a molecule

```python
from solvaspec import predict_solvent_spectrum

result = predict_solvent_spectrum(
    smiles="CC(C)(C)c1ccc(cc1)/C(O)=C\\C(=O)c1ccc(OC)cc1",  # avobenzone enol
    solvent="ethanol",                                       # solvent SMILES or name
)
print(f"lambda_max = {result.lambda_max_nm:.1f} nm")
print(f"epsilon_max = {result.eps_max:.2e} M^-1 cm^-1")
print(f"UVA integrated absorbance = {result.uva:.3f}")
print(f"UVB integrated absorbance = {result.uvb:.3f}")
result.plot()  # saves a PNG of the predicted curve
```

## Data archives

| Resource | DOI |
|---|---|
| Source code (this repo) | Zenodo: `10.5281/zenodo.XXXXXXXX` (assigned on first release) |
| Wetlab UV-Vis measurements (32 spectra) | Zenodo: `10.5281/zenodo.YYYYYYYY` |
| F13 solute-solvent corpus splits | Zenodo: `10.5281/zenodo.ZZZZZZZZ` |
| ε calibration set indices | Zenodo: `10.5281/zenodo.WWWWWWWW` |

The Zenodo entries are the canonical citation targets. This GitHub repo is the development mirror; cite the Zenodo DOIs in academic work.

## Reproducing every paper figure

```bash
python -m scripts.regenerate_all_figures
```

This sequentially regenerates every figure in `figures/` from the underlying CSV/JSON data. Expected wallclock: ~15 minutes on a CPU.

## Citation

```bibtex
@article{Arampath2026SolvaSpec,
  title   = {SolvaSpec: An experimentally validated deep-learning prediction
             platform for solvent-aware UV-Vis spectra and extinction coefficients},
  author  = {Arampath, Umesh and Pioch, Birgit and Detrich, David},
  journal = {npj Computational Materials},
  year    = {2026},
  note    = {In submission}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Contact

Umesh Arampath — `umeshcmbac@gmail.com`
Midwest Bioprocessing Center, IL, USA
