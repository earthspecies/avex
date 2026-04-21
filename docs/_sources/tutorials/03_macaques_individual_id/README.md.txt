# Macaques Individual ID

Identifies which of 8 macaque individuals produced a coo call, and classifies caller sex, using avex embeddings.

**Inspired by:** [ESP library / macaques](https://github.com/earthspecies/library/tree/main/macaques)

## Dataset

- **Species:** Japanese macaque (*Macaca fuscata*)
- **Size:** 7,285 coo calls, 8 individuals (4 male / 4 female)
- **Sample rates:** 24,414 Hz and 44,100 Hz (resampled to 16 kHz)
- **Source:** Internet Archive

## Notebooks

| Notebook | Description |
|---|---|
| `macaques_individual_id.ipynb` | End-to-end workflow: download → explore → embed → UMAP → training-free metrics (NMI/ARI/R-AUC) → probes → save `artifacts/` |
| `macaques_individual_id_with_results.ipynb` | Executed notebook with rendered outputs (generated via `run_notebooks.py`) |

Legacy stepwise notebooks may still exist in the folder, but `macaques_individual_id.ipynb` is the canonical, maintained version.

## Key results

BEATs and EfficientNet embeddings are compared on two tasks:
1. **Individual identification** (8 classes)
2. **Sex classification** (2 classes)
