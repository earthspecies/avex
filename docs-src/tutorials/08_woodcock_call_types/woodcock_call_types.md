# Scripts & Configuration

End-to-end call type classification for Eurasian woodcock (*Scolopax rusticola*) using passive acoustic monitoring data from Germany.

## Dataset

- **Source:** [Zenodo record 10930931](https://zenodo.org/records/10930931)
- **Classes:** 3 call types (whistle, croak, chase)
- **Size:** 2,545 labelled 3-second clips from 9 deployments

| Type | Original annotations | Clips | Share |
|------|---------------------|-------|-------|
| whistle | whistle, woodcock | ~2018 | ~79% |
| croak | croak | ~464 | ~18% |
| chase | chase | ~24 | ~1% |

## Pipeline

```
audio files ──► embedding extraction ──► probe training ──► random split + LODO CV
```

Both avex models are compared:

- `esp_aves2_sl_beats_all` — last-layer and all-layers average
- `esp_aves2_effnetb0_all` — global average pool

## Usage

```bash
cd examples/08_woodcock_call_types

# Train probes and save embeddings
python train.py --config config.yaml

# Evaluate with LODO cross-validation
python evaluate.py --config config.yaml --model esp_aves2_sl_beats_all
```

## Design Notes

- **Balanced accuracy** is used throughout because the class distribution is severely skewed (~79% whistle).
- **Leave-one-deployment-out (LODO)** cross-validation tests generalisation to unseen recording sites — a more realistic metric than a random split.
- The `chase` class is extremely rare; per-class recall shows how well each model handles it.
