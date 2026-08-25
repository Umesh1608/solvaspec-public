# Data archives

Large datasets are not committed to git. Download from Zenodo and place in
this directory.

| File | Description | Source |
|---|---|---|
| `wetlab_32_spectra.csv` | 32 in-house UV-Vis measurements (16 FDA filters × 2 solvents) | Zenodo `10.5281/zenodo.XXXX` |
| `wetlab_lambda_max.csv` | Derived λ_max for each of 32 measurements (16 FDA filters × 2 solvents) | **Bundled in this repo** |
| `solute_solvent_pairs.csv` | 26,487 solute–solvent pairs with experimental λ_max | Zenodo `10.5281/zenodo.XXXX` |
| `solute_solvent_splits.json` | Train/val/test indices for the five-fold solute-stratified split | (same archive) |
| `eps_calibration_set.csv` | 6,283 (molecule, solvent, ε_max) pairs from PhotochemCAD + Joung | Zenodo `10.5281/zenodo.XXXX` |
| `eps_splits.json` | 80/10/10 split indices for the ε corpus | (same archive) |
| `ornl_aisd_ex_filter_indices.json` | Indices of the 5.26M molecules retained from the 10.5M ORNL_AISD-Ex corpus | Zenodo `10.5281/zenodo.XXXX` |

## Source databases

The upstream public databases are NOT redistributed here; obtain them from
the original sources:

- **ORNL_AISD-Ex**: https://constellation.ornl.gov (Lupo Pasini et al., 2023)
- **GDB-9-Ex**: https://constellation.ornl.gov (Lupo Pasini et al., 2023)
- **QM9S**: https://github.com/Zhonghui-Wu/DetaNet/tree/main/qm9s (Zhong et al., 2023)
- **PhotochemCAD**: https://www.photochemcad.com/ (Du et al., 1998)
- **Joung optical properties database**: as released by Joung et al., 2020
- **Beard photochemistry compilation**: as released by Beard et al., 2019
- **Mamede UV database**: as released by Mamede et al., 2021

## Auto-download (the slices we use)

```bash
python -m solvaspec.download_data
```
