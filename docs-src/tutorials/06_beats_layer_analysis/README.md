# BEATs Layer Analysis

Probes each transformer layer of `esp_aves2_sl_beats_all` to show how audio representations evolve from low-level acoustics to high-level semantics. Compares last-layer embeddings against a concatenation of all layers.

## Motivation

Transformer models build progressively richer representations across layers. This notebook makes that progression visible: a linear probe trained at each layer reveals when task-relevant information is encoded, and whether aggregating all layers outperforms using only the final output.

## Method

1. Load `esp_aves2_sl_beats_all` with `return_features_only=True`
2. Register `register_forward_hook` on each transformer block to capture hidden states
3. Run a single forward pass — all layers captured simultaneously
4. For each layer: mean-pool token sequence → train linear probe → record accuracy
5. Compare last-layer vs concatenation of all layers

## Datasets used

- **Giant Otter** (21 call types) — from `examples/01_giant_otter_classifier/`
- **Zebra Finch** (11 call types) — from `examples/05_zebra_finch/`

Run those examples first to generate embeddings, or use the synthetic-data fallback included in the notebook.

## Outputs

| Figure | Description |
|---|---|
| Probing accuracy curve | Accuracy at each BEATs layer for both datasets |
| UMAP grid | Layer 0 vs layer 6 vs last layer, side by side |
| Last vs all comparison | Bar chart: last-layer probe vs all-layers concatenated |
