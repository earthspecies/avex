# AVEX - Animal Vocalization Encoder Library

![CI status](https://github.com/earthspecies/avex/actions/workflows/pythonapp.yml/badge.svg?branch=main)
![Pre-commit status](https://github.com/earthspecies/avex/actions/workflows/pre-commit.yml/badge.svg?branch=main)

An API for model loading and inference, and a Python-based system for training and evaluating bioacoustics representation learning models.

## Description

The Animal Vocalization Encoder library AVEX provides a unified interface for working with pre-trained bioacoustics representation learning models, with support for:

- **Model Loading**: Load pre-trained models with checkpoints and class mappings
- **Embedding Extraction**: Extract features from audio for downstream tasks
- **Probe System**: Flexible probe heads (linear, MLP, LSTM, attention, transformer) for transfer learning
- **Training & Evaluation**: Scripts for supervised learning experiments
- **Plugin Architecture**: Register and use custom models seamlessly

## Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12

### Install with pip

```bash
pip install avex
```

### Install with uv

```bash
uv add avex
```

For development installation with training/evaluation tools, see the [Contributing guide](CONTRIBUTING.md).

## Quick Start

```python
import torch
import librosa
from avex import load_model, list_models

# List available models
print(list_models().keys())

# Load a pre-trained model
model = load_model("esp_aves2_sl_beats_all", device="cpu")

# Load and preprocess audio (BEATs expects 16kHz)
audio, sr = librosa.load("your_audio.wav", sr=16000)
audio_tensor = torch.tensor(audio).unsqueeze(0)  # Shape: (1, num_samples)

# Run inference
with torch.no_grad():
    logits = model(audio_tensor)
    predicted_class = logits.argmax(dim=-1).item()

# Get human-readable label
if model.label_mapping:
    label = model.label_mapping.get(str(predicted_class), predicted_class)
    print(f"Predicted: {label}")
```

### Embedding Extraction

```python
# Load for embedding extraction (no classifier head)
model = load_model("esp_aves2_sl_beats_all", return_features_only=True, device="cpu")

with torch.no_grad():
    embeddings = model(audio_tensor)
    # Shape: (batch, time_steps, 768) for BEATs

# Pool to get fixed-size embedding
embedding = embeddings.mean(dim=1)  # Shape: (batch, 768)
```

### Transfer Learning with Probes

```python
from avex.models.probes import build_probe_from_config
from avex.configs import ProbeConfig

# Load backbone for feature extraction
base = load_model("esp_aves2_sl_beats_all", return_features_only=True, device="cpu")

# Define a probe head for your task
probe_config = ProbeConfig(
    probe_type="linear",
    target_layers=["last_layer"],
    aggregation="mean",
    freeze_backbone=True,
    online_training=True,
)

probe = build_probe_from_config(
    probe_config=probe_config,
    base_model=base,
    num_classes=10,  # Your number of classes
    device="cpu",
)
```

## Documentation

**Full documentation**: [docs/index.md](docs/index.md)

### Core Documentation

- **[API Reference](docs/api_reference.md)** - Complete API documentation for model loading, registry, and management functions
- **[Architecture](docs/api_architecture.md)** - Framework architecture, core components, and plugin system
- **[Supported Models](docs/supported_models.md)** - List of supported models and their configurations
- **[Configuration](docs/configuration.md)** - ModelSpec parameters, audio requirements, and configuration options

### Usage Guides

- **[Training and Evaluation](docs/training_evaluation.md)** - Guide to training and evaluating models
- **[Embedding Extraction](docs/embedding_extraction.md)** - Working with feature representations and embeddings
- **[Examples](docs/examples.md)** - Comprehensive examples and use cases

### Advanced Topics

- **[Probe System](docs/probe_system.md)** - Understanding and using probes for transfer learning
- **[API Probes](docs/api_probes.md)** - API reference for probe-related functionality
- **[Custom Model Registration](docs/custom_model_registration.md)** - Guide on registering custom model classes and loading pre-trained models

**Examples**: See the [`examples/`](examples/) directory:

- `00_quick_start.py` - Basic model loading
- `01_basic_model_loading.py` - Loading models with different configurations
- `02_checkpoint_loading.py` - Working with checkpoints
- `03_custom_model_registration.py` - Custom model registration
- `04_training_and_evaluation.py` - Training and evaluation examples
- `05_embedding_extraction.py` - Feature extraction
- `06_classifier_head_loading.py` - Classifier head behavior

## Supported Models

The framework supports the following audio representation learning models:

- **EfficientNet** - EfficientNet-based models for audio classification
- **BEATs** - BEATs transformer models for audio representation learning
- **EAT** - Efficient Audio Transformer models
- **AVES** - AVES model for bioacoustics
- **BirdMAE** - BirdMAE masked autoencoder for bioacoustic representation learning
- **ATST** - Audio Spectrogram Transformer
- **ResNet** - ResNet models (ResNet18, ResNet50, ResNet152)
- **CLIP** - Contrastive Language-Audio Pretraining models
- **BirdNet** - BirdNet models for bioacoustic classification
- **Perch** - Perch models for bioacoustics
- **SurfPerch** - SurfPerch models

See [Supported Models](docs/supported_models.md) for detailed information and configuration examples.

## Supported Probes

The framework provides flexible probe heads for transfer learning:

- **Linear** - Simple linear classifier (fastest, most memory-efficient)
- **MLP** - Multi-layer perceptron with configurable hidden layers
- **LSTM** - Long Short-Term Memory network for sequence modeling
- **Attention** - Self-attention mechanism for sequence modeling
- **Transformer** - Full transformer encoder architecture

Probes can be trained:
- **Online**: End-to-end with the backbone (raw audio input)
- **Offline**: On pre-computed embeddings

See [Probe System](docs/probe_system.md) and [API Probes](docs/api_probes.md) for detailed documentation.

## Citing

If you use this framework in your research, please cite:

```bibtex
@article{miron2025matters,
  title={What Matters for Bioacoustic Encoding},
  author={Miron, Marius and Robinson, David and Alizadeh, Milad and Gilsenan-McMahon, Ellen and Narula, Gagan and Pietquin, Olivier and Geist, Matthieu and Chemla, Emmanuel and Cusimano, Maddie and Effenberger, Felix and others},
  journal={arXiv preprint arXiv:2508.11845},
  year={2025}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Running tests
- Code style guidelines
- Adding new functionality
- Pull request process

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of PyTorch
- Integrates with various pre-trained audio models
