# Supported Models

## Official Models

The framework includes support for various audio representation learning models:

- **EfficientNet**: EfficientNet-based models adapted for audio classification
- **BEATs**: BEATs transformer models for audio representation learning
- **EAT**: Efficient Audio Transformer models
- **AVES**: AVES model for bioacoustics
- **BirdMAE**: BirdMAE masked autoencoder for bioacoustic representation learning

### Labels vs features only

| Capability | Description |
|------------|-------------|
| **Classification with labels** | Model has a trained classifier head and a class mapping (e.g. `label_map.json`). Use `load_model("model_name", device="cpu")` to get logits and use `model.label_mapping` for human-readable class names. |
| **Features / embeddings only** | Any model can be loaded for embedding extraction by passing `return_features_only=True`. The model then returns feature tensors instead of classification logits. |

**How to see which models offer what**

- **At runtime**: Call `list_models()` — the printed table has a "Trained Classifier" column (✅ = has checkpoint + class mapping, ❌ = backbone/features only). The returned dict includes `has_trained_classifier` and `num_classes` per model.
- **Per model**: Call `describe_model("model_name", verbose=True)` to see "Has Trained Classifier", checkpoint path, class mapping path, and number of classes.

**Current official models (ESP-AVES2)**
All official models in `representation_learning/api/configs/official_models/` (e.g. `esp_aves2_sl_beats_all`, `esp_aves2_effnetb0_all`) currently have both a checkpoint and a class mapping, so they support **classification with labels**. They also support **embedding extraction** with `load_model(..., return_features_only=True)`.

## Model Configuration

Models are configured using YAML files which contain the model specifications `model_spec`. The official config files are in the `representation_learning/api/configs/official_models/` directory. These files define the model architecture, audio preprocessing parameters, and optional checkpoint/label mapping paths.

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
class_mapping_path: gs://my-bucket/models/label_map.json

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
