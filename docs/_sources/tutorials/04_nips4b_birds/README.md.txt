# 04 — NIPS4B Bird Species

Multi-label bird species detection using the NIPS4B 2013 bird challenge dataset.

**Dataset**: 687 recordings of French bird songs (44 100 Hz, ~5 s each), 87 species
**Labels**: Official NIPS4B challenge binary labels (`numero_file_train.csv`)
**Task**: Per-species binary detection (species present vs absent)

## Notebooks

| Notebook | Description |
|---|---|
| `nips4b_birds.ipynb` | End-to-end workflow: download → explore → embed → UMAP → training-free metrics (NMI/ARI/R-AUC) → per-species probes → save `artifacts/` |
| `nips4b_birds_with_results.ipynb` | Executed notebook with rendered outputs (generated via `run_notebooks.py`) |

Legacy stepwise notebooks may still exist in the folder, but `nips4b_birds.ipynb` is the canonical, maintained version.

## Data sources

- Audio: `http://sabiod.univ-tln.fr/nips4b/media/birds/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV.tar.gz`
- Labels: `http://sabiod.univ-tln.fr/nips4b/media/birds/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS.tar`
- Challenge: [NIPS4B 2013 Bird Song Classification](http://sabiod.univ-tln.fr/nips4b/)
