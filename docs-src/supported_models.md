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

These configurations can be loaded directly with `load_model("path/to/config.yml")`. See the {doc}`Custom Model Registration <custom_model_registration>` section for usage examples.
