# Ungulate Emotional Valence

Classify emotional valence (positive vs. negative) and species from ungulate contact calls using avex embeddings.

**Dataset:** [Zenodo 14636641](https://zenodo.org/records/14636641) — supplemental data for *Machine Learning Algorithms Can Predict Emotional Valence Across Ungulate Vocalizations*

## Dataset

- **Species:** Cow, Pig, Sheep, Goat, Horse, Przewalski's Horse, Wild Boar (7 species)
- **Size:** 3,181 contact call recordings
- **Labels:** emotional valence (positive / negative)
- **Source:** Zenodo (CC BY 4.0)

## Notebooks

| Notebook | Description |
|---|---|
| `ungulate_valence.ipynb` | End-to-end workflow: download → explore → embed → UMAP → training-free metrics (NMI/ARI/R-AUC) → linear probe → attention probe → cross-species LOSO → save `artifacts/` |
| `ungulate_valence_with_results.ipynb` | Executed notebook with rendered outputs (generated via `run_notebooks.py`) |

## Key results

BEATs and EfficientNet embeddings are compared on two tasks:
1. **Valence classification** (2 classes: positive / negative)
2. **Species identification** (7 classes)

### Cross-species evaluation (LOSO)

Leave-one-species-out probe: train on 6 species, test on the held-out 7th.
Measures transfer to an unseen species at inference time.

## Annotations

Raw annotations from the Zenodo Excel spreadsheet are converted once via `openpyxl` and stored as `artifacts/annotations.csv` — subsequent runs load from the CSV directly.
