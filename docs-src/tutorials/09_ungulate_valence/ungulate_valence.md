# Ungulate Emotional Valence

Classify emotional valence (positive vs. negative) and species from ungulate contact calls using avex embeddings.

**Dataset:** [Zenodo 14636641](https://zenodo.org/records/14636641) — supplemental data for *Machine Learning Algorithms Can Predict Emotional Valence Across Ungulate Vocalizations*

## Dataset

- **Species:** Cow, Pig, Sheep, Goat, Horse, Przewalski's Horse, Wild Boar (7 species)
- **Size:** 3,181 contact call recordings
- **Labels:** emotional valence (positive / negative)
- **Source:** Zenodo (CC BY 4.0)

## Pipeline

```
download dataset ──► explore + annotate ──► embed (BEATs / EfficientNet)
    ──► UMAP ──► training-free metrics ──► linear probe
    ──► attention probe ──► LOSO cross-species eval ──► speed augmentation
```

## Key results

BEATs and EfficientNet embeddings are compared on two tasks:

1. **Valence classification** (positive / negative)
2. **Species identification** (7 classes)

### Cross-species evaluation (LOSO)

Leave-one-species-out probe: train on 6 species, test on the held-out 7th.
Measures how well embeddings transfer to an unseen species.

### Speed augmentation

Training set augmented with `librosa.effects.time_stretch` at 14 rates (0.5× – 2.0×).
Tests whether temporal-rate invariance improves cross-species generalisation.
