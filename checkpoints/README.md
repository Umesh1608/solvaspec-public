# Model checkpoints

Checkpoints are not committed to git (they are 6–12 MB each and binary).
Download from the Zenodo archive and place them here:

| File | Source | Size |
|---|---|---|
| `painn_sca_50exc_oschead_3layer.pt` | Zenodo `10.5281/zenodo.XXXXXXXX` | 6.1 MB |
| `spectrum_hybrid_5fold.pt` | Zenodo `10.5281/zenodo.XXXXXXXX` | 1.2 MB (ensemble) |
| `eps_random_forest.joblib` | Zenodo `10.5281/zenodo.XXXXXXXX` | 8.3 MB |

## Auto-download

```bash
python -m solvaspec.download_checkpoints
```

Downloads all three artefacts via `gdown` and places them in this directory.

## Verifying the download

After running the download script, run:

```bash
python -c "from solvaspec import verify_checkpoints; verify_checkpoints()"
```

This loads each checkpoint, runs a single-molecule sanity inference, and
prints `PASS` or `FAIL`.
