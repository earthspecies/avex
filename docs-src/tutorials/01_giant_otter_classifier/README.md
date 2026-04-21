# Giant Otter Classifier

End-to-end classification pipeline for 21 giant otter (*Pteronura brasiliensis*) call types using avex transfer learning.

**Inspired by:** [ESP library / giant_otter](https://github.com/earthspecies/library/blob/main/giant_otter/cnn-classifier-pipeline.ipynb)

## Dataset

- **Source:** Internet Archive
- **Classes:** 21 call types (barks, screams, contact calls, whistles, humming, ...)
- **Size:** 9–32 recordings per class

## Pipeline

```
audio files → embedding extraction → linear probe → accuracy + confusion matrix
```

Two models are compared side by side:
- `esp_aves2_sl_beats_all` — mean pool over temporal tokens
- `esp_aves2_effnetb0_all` — global average pool over spatial maps

## Usage

```bash
# Train probes for both models
python train.py --config config.yaml

# Evaluate one model and save confusion matrix
python evaluate.py --config config.yaml --model esp_aves2_sl_beats_all
```

## Outputs

| File | Description |
|---|---|
| `examples/01_giant_otter_classifier/data/embeddings/*.npy` | Cached embedding arrays (not committed) |
| `examples/01_giant_otter_classifier/artifacts/umap_*.html` | Interactive UMAP scatter (committed) |
| `examples/01_giant_otter_classifier/artifacts/model_comparison.html` | Accuracy bar chart (committed) |
| `examples/01_giant_otter_classifier/artifacts/confusion_*.html` | Confusion matrix heatmap (committed) |
| `examples/01_giant_otter_classifier/artifacts/giant_otter_metrics.json` | Metrics JSON incl. training-free metrics (committed) |
