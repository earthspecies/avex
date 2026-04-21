# Interactive Visualization

Loads embeddings from any upstream avex-examples pipeline and produces an interactive UMAP dashboard comparing BEATs and EfficientNet embedding spaces side by side.

**Inspired by:** [bacpipe](https://github.com/bioacoustic-ai/bacpipe) visualization dashboard

## Usage

1. Point `embeddings_dir` in the notebook config cell at a folder containing `.npy` or `.pt` embedding files (e.g. output from `examples/02_embed_audio/`).
2. Optionally provide a `labels_file` CSV with `filename` and `label` columns.
3. Run all cells — a self-contained `embedding_explorer.html` is exported.

If no embeddings are found the notebook falls back to synthetic data so it runs end-to-end out of the box.

## Features

- Side-by-side BEATs vs EfficientNet UMAP panels
- Points colored by class label
- Hover shows filename and label
- Exports as self-contained HTML (no server required)
