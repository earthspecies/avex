# 08 — Eurasian Woodcock Call Types

Call type classification for Eurasian woodcock (*Scolopax rusticola*) using passive acoustic
monitoring data from Germany.

**Dataset**: 2,545 labelled 3-second clips from 9 deployments, 3 simplified call types
**Labels**: Raven Pro selection tables, simplified from 4 original annotations
**Task**: 3-way call type classification + cross-site generalisation (LODO)

## Call types

| Type | Original annotations | Clips | Share |
|------|---------------------|-------|-------|
| whistle | whistle, woodcock | ~2018 | ~79 % |
| croak | croak | ~464 | ~18 % |
| chase | chase | ~24 | ~1 % |

The `whistle` class merges the "woodcock" annotation (species present, tonal calls) and
explicit "whistle" labels; "croak" is the grunting display call; "chase" is a rapid
flight-pursuit call type.

## Notebooks

| Notebook | Description |
|---|---|
| `woodcock_call_types.ipynb` | End-to-end workflow: download → explore → embed (incl. BEATs all-layers avg baseline) → UMAP → training-free metrics (NMI/ARI/R-AUC) → probes (random + LODO) → save `artifacts/` |
| `woodcock_call_types_with_results.ipynb` | Executed notebook with rendered outputs (generated via `run_notebooks.py`) |

Legacy stepwise notebooks may still exist in the folder, but `woodcock_call_types.ipynb` is the canonical, maintained version.

## Data sources

- Dataset: [Zenodo record 10930931](https://zenodo.org/records/10930931)
- Audio ZIP: `https://zenodo.org/records/10930931/files/selections_wavs.zip`
- Annotations: `https://zenodo.org/records/10930931/files/selections.csv`

## Design notes

- **Balanced accuracy** is used throughout because the class distribution is severely skewed
  (~79 % whistle).
- **Leave-one-deployment-out (LODO)** cross-validation tests generalisation to unseen recording
  sites — a more realistic metric than a random split when clips from the same deployment appear
  in both train and test.
- The `chase` class is extremely rare; per-class recall shows how well each model handles it.
