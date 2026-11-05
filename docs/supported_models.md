# Supported Models

## Official Models (ESP-AVES2)

The ESP-AVES2 model collection is available on HuggingFace: [EarthSpeciesProject/esp-aves2](https://huggingface.co/collections/EarthSpeciesProject/esp-aves2)

### Available Models

| Model Name | Architecture | Training Data | HuggingFace |
|------------|--------------|---------------|-------------|
| `esp_aves2_sl_beats_all` | BEATs | All (AudioSet + Bio) | [Link](https://huggingface.co/EarthSpeciesProject/esp-aves2-sl-beats-all) |
| `esp_aves2_sl_beats_bio` | BEATs | Bioacoustics | [Link](https://huggingface.co/EarthSpeciesProject/esp-aves2-sl-beats-bio) |
| `esp_aves2_naturelm_audio_v1_beats` | BEATs + NatureLM | All | [Link](https://huggingface.co/EarthSpeciesProject/esp-aves2-naturelm-audio-v1-beats) |
| `esp_aves2_eat_all` | EAT | All (AudioSet + Bio) | [Link](https://huggingface.co/EarthSpeciesProject/esp-aves2-eat-all) |
| `esp_aves2_eat_bio` | EAT | Bioacoustics | [Link](https://huggingface.co/EarthSpeciesProject/esp-aves2-eat-bio) |
| `esp_aves2_sl_eat_all_ssl_all` | EAT (SSL) | All | [Link](https://huggingface.co/EarthSpeciesProject/esp-aves2-sl-eat-all-ssl-all) |
| `esp_aves2_sl_eat_bio_ssl_all` | EAT (SSL) | Bioacoustics | [Link](https://huggingface.co/EarthSpeciesProject/esp-aves2-sl-eat-bio-ssl-all) |
| `esp_aves2_effnetb0_all` | EfficientNet-B0 | All (AudioSet + Bio) | [Link](https://huggingface.co/EarthSpeciesProject/esp-aves2-effnetb0-all) |
| `esp_aves2_effnetb0_bio` | EfficientNet-B0 | Bioacoustics | [Link](https://huggingface.co/EarthSpeciesProject/esp-aves2-effnetb0-bio) |
| `esp_aves2_effnetb0_audioset` | EfficientNet-B0 | AudioSet | [Link](https://huggingface.co/EarthSpeciesProject/esp-aves2-effnetb0-audioset) |

### Supported Architectures

- **BEATs**: Bidirectional Encoder representation from Audio Transformers
- **EAT**: Efficient Audio Transformer models
- **EfficientNet**: EfficientNet-based models adapted for audio classification
- **AVES**: AVES model for bioacoustics
- **BirdMAE**: BirdMAE masked autoencoder for bioacoustic representation learning

### Labels vs Features Only

| Capability | Description |
|------------|-------------|
| **Classification with labels** | Model has a trained classifier head and a class mapping (e.g. `label_map.json`). Use `load_model("model_name", device="cpu")` to get logits and use `model.label_mapping` for human-readable class names. |
| **Features / embeddings only** | Any model can be loaded for embedding extraction by passing `return_features_only=True`. The model then returns feature tensors instead of classification logits. |

**How to see which models offer what**

- **At runtime**: Call `list_models()` — the printed table has a "Trained Classifier" column (✅ = has checkpoint + class mapping, ❌ = backbone/features only). The returned dict includes `has_trained_classifier` and `num_classes` per model.
- **Per model**: Call `describe_model("model_name", verbose=True)` to see "Has Trained Classifier", checkpoint path, class mapping path, and number of classes.

All official ESP-AVES2 models have both a checkpoint and a class mapping, so they support **classification with labels**. They also support **embedding extraction** with `load_model(..., return_features_only=True)`.

## Upstream BEATs Backbones (bring your own checkpoint)

The two Microsoft BEATs checkpoints that the ESP-AVES2 BEATs models were trained from are
registered so they can be loaded through the same API, but their weights are **not bundled**
and are not mirrored on the Hugging Face Hub. Download them from the
[upstream BEATs release](https://github.com/microsoft/unilm/tree/master/beats) and pass the
file via `checkpoint_path`:

| Model Name | Description | Upstream file |
|------------|-------------|---------------|
| `beats_iter3_plus_as2m_ssl` | BEATs iter3+, self-supervised on AudioSet-2M (SSL backbone) | `BEATs_iter3_plus_AS2M.pt` |
| `beats_iter3_plus_as2m_finetuned_cpt2` | BEATs iter3+, SSL then fine-tuned on AudioSet-2M (checkpoint "cpt2") | `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` |

```python
from avex import load_model

model = load_model(
    "beats_iter3_plus_as2m_ssl",
    checkpoint_path="/path/to/BEATs_iter3_plus_AS2M.pt",  # local file or gs://, hf://, ... URI
    device="cpu",
)
features = model(audio)  # (batch, frames, 768)
```

The upstream `.pt` files are accepted as-is. Both models are backbones only, so they always load
in embedding-extraction mode (the fine-tuned checkpoint's 527-class AudioSet predictor is loaded
into the backbone but is not exposed as a classifier). Calling `load_model` on either name
without `checkpoint_path` raises a `ValueError` rather than silently returning a randomly
initialised network.

## Model Configuration

Models are configured using YAML files which contain the model specifications `model_spec`. The official config files are in the `avex/api/configs/official_models/` directory. These files define the model architecture, audio preprocessing parameters, and optional checkpoint/label mapping paths.

**Minimal Model Configuration:**

```yaml
# Example: my_model.yml - Minimal configuration for model loading
model_spec:
  name: efficientnet
  pretrained: false
  device: cuda
  audio_config:
    sample_rate: 16000
    representation: mel_spectrogram
    n_mels: 128
  efficientnet_variant: b0
```

**Full Model Configuration (with checkpoint):**

```yaml
# Example: esp_aves2_effnetb0_all.yml - Complete configuration
# Optional: Default checkpoint path
checkpoint_path: hf://EarthSpeciesProject/esp-aves2-effnetb0-all/esp-aves2-effnetb0-all.safetensors

# Optional: Label mapping for human-readable predictions
class_mapping_path: hf://EarthSpeciesProject/esp-aves2-effnetb0-all/label_map.json

# Required: Model specification
model_spec:
  name: efficientnet
  pretrained: false
  device: cuda
  audio_config:
    sample_rate: 16000
    representation: mel_spectrogram
    n_mels: 128
    target_length_seconds: 10
  efficientnet_variant: b0
```

These configurations can be loaded directly with `load_model("path/to/config.yml")`. See the [Custom Model Registration](custom_model_registration.md) section for usage examples.

## External Models

### ONNX-based (no extra dependencies)

| Model name | Description | Source |
|------------|-------------|--------|
| `birdnet` | BirdNet v2.4 — 6 522 species, 1 024-d embeddings, 48 kHz | [justinchuby/BirdNET-onnx](https://huggingface.co/justinchuby/BirdNET-onnx) |
| `perch2` | Google Perch v2 — 14 795 species, 1 536-d embeddings, 32 kHz | [justinchuby/Perch-onnx](https://huggingface.co/justinchuby/Perch-onnx) |

### TensorFlow-based *(deprecated)*

The following models use TensorFlow Hub and require an optional extra:

```bash
pip install 'avex[tensorflow]'
```

| Model name | Description | Status |
|------------|-------------|--------|
| `perch` | Google Perch v1 — 10 932 species, 1 280-d embeddings, 32 kHz | Deprecated — migrate to `perch2` |
| `surfperch` | SurfPerch via TensorFlow Hub | Deprecated — no ONNX export available yet |

These models emit a `DeprecationWarning` on instantiation. TensorFlow support will be removed in a future release.
