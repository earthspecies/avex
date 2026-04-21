# Usage

Generic batch embedding extraction for any directory of `.wav` or `.flac` files.

**Inspired by:** [Perch / embed_audio.ipynb](https://github.com/google-research/perch/blob/main/embed_audio.ipynb)

## Usage

```bash
cd examples/02_embed_audio
python embed_audio.py --config config.yaml
```

## What it does

1. Scans `audio.input_dir` for `.wav` / `.flac` files
2. Shards each file into fixed-length overlapping windows
3. Runs both avex models in `return_features_only=True` mode
4. Mean-pools the output tokens / spatial maps to a 1-D vector per window
5. Saves one `.pt` file per audio file per model (shape `(n_windows, dim)`)

Already-extracted files are skipped — the script is idempotent.

## Configuration

```yaml
audio:
  input_dir: "data/audio"
  sample_rate: 16000
  window_seconds: 5.0
  hop_seconds: 2.5

models:
  - name: "esp_aves2_sl_beats_all"
  - name: "esp_aves2_effnetb0_all"

output:
  embeddings_dir: "outputs/embeddings"
```

## Output format

```
outputs/embeddings/
├── esp_aves2_sl_beats_all/
│   ├── recording_001.pt   # shape: (n_windows, 768)
│   └── recording_002.pt
└── esp_aves2_effnetb0_all/
    ├── recording_001.pt   # shape: (n_windows, C)
    └── recording_002.pt
```

:::{tip}
The embeddings produced here are the expected input format for
{doc}`../../examples/07_interactive_visualization/visualize`.
:::
