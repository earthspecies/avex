# Supported Models

## Official Models

The framework includes support for various audio representation learning models:

- **EfficientNet**: EfficientNet-based models adapted for audio classification
- **BEATs**: BEATs transformer models for audio representation learning
- **EAT**: Efficient Audio Transformer models
- **AVES**: AVES model for bioacoustics
- **BirdMAE**: BirdMAE masked autoencoder for bioacoustic representation learning

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
# Example: efficientnet_animalspeak.yml - Complete configuration
# Optional: Default checkpoint path
checkpoint_path: gs://my-bucket/models/efficientnet_animalspeak.pt

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
