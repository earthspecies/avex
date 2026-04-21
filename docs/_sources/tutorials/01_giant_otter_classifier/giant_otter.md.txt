# Usage

End-to-end classification pipeline for 21 giant otter (*Pteronura brasiliensis*) call types using avex transfer learning.

**Inspired by:** [ESP library / giant_otter](https://github.com/earthspecies/library/blob/main/giant_otter/cnn-classifier-pipeline.ipynb)

## Dataset

- **Source:** Internet Archive
- **Classes:** 21 call types (barks, screams, contact calls, whistles, humming, …)
- **Size:** 9–32 recordings per class

## Pipeline

```
audio files ──► embedding extraction ──► linear probe ──► accuracy + figures
```

Both avex models are compared:

- `esp_aves2_sl_beats_all` — mean pool over temporal tokens `(N, T, 768) → (N, 768)`
- `esp_aves2_effnetb0_all` — global average pool `(N, C, H, W) → (N, C)`

## Usage

```bash
cd examples/01_giant_otter_classifier

# Train probes for both models, save embeddings + UMAP figure
python train.py --config config.yaml

# Evaluate one model and save confusion matrix
python evaluate.py --config config.yaml --model esp_aves2_sl_beats_all
```

## Configuration

Key fields in `config.yaml`:

```yaml
dataset:
  url: "https://archive.org/..."
  sample_rate: 16000
  window_seconds: 3.0

models:
  - name: "esp_aves2_sl_beats_all"
    pooling: "mean"   # mean | cls | max
  - name: "esp_aves2_effnetb0_all"
    pooling: "mean"

probe:
  type: "linear"
  test_size: 0.2
```

## Outputs

| File | Description |
|---|---|
| `outputs/.../embeddings/*.npy` | Per-model embedding arrays |
| `outputs/.../figures/umap_*.html` | Interactive UMAP scatter (Plotly) |
| `outputs/.../figures/model_comparison.html` | Accuracy bar chart |
| `outputs/.../figures/confusion_*.html` | Normalised confusion matrix |

:::{note}
Embeddings saved here are used as input to {doc}`../06_beats_layer_analysis/beats_layer_analysis_with_results`.
:::
