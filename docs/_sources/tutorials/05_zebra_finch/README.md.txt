# Zebra Finch

Classifies 11 zebra finch (*Taeniopygia guttata*) call types and distinguishes adult from juvenile callers using avex embeddings.

**Inspired by:** [ESP library / zebra_finch](https://github.com/earthspecies/library/tree/main/zebra_finch)

## Dataset

- **Source:** Figshare — collected by Julie E. Elie (2011–2014) at UC Berkeley Theunissen Lab
- **Size:** 3,433 calls, 11 call types
- **Callers:** Adult birds + juvenile chicks (~30 days old)
- **Sample rate:** 44,100 Hz

## Notebooks

| Notebook | Description |
|---|---|
| `zebra_finch.ipynb` | End-to-end workflow: download → explore → embed → UMAP → training-free metrics (NMI/ARI/R-AUC) → probes → save `artifacts/` |
| `zebra_finch_with_results.ipynb` | Executed notebook with rendered outputs (generated via `run_notebooks.py`) |

Legacy stepwise notebooks may still exist in the folder, but `zebra_finch.ipynb` is the canonical, maintained version.

## Notes

Embeddings from this example are used in `06_beats_layer_analysis/` for cross-dataset comparison.
