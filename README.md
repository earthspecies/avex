# Representation Learning Framework

An API for model loading and inference, and a Python-based system for training and evaluating bioacoustics representation learning models.

## Description

The Representation Learning Framework provides a unified interface for working with pre-trained bioacoustics representation learning models, with support for:

- **Model Loading**: Load pre-trained models with checkpoints and class mappings
- **Embedding Extraction**: Extract features from audio for downstream tasks
- **Probe System**: Flexible probe heads (linear, MLP, LSTM, attention, transformer) for transfer learning
- **Training & Evaluation**: Scripts for supervised learning experiments
- **Plugin Architecture**: Register and use custom models seamlessly

## Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12

### Install with uv (Recommended)

```bash
# 1. Install keyring with Google Artifact Registry plugin
uv tool install keyring --with keyrings.google-artifactregistry-auth

# 2. Create and activate virtual environment
uv venv
source .venv/bin/activate

# 3. Configure uv (add to pyproject.toml)
[[tool.uv.index]]
name = "esp-pypi"
url = "https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/"
explicit = true

[tool.uv.sources]
representation-learning = { index = "esp-pypi" }

[tool.uv]
keyring-provider = "subprocess"

# 4. Install the package
uv add representation-learning
```

### Install with pip

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install from ESP index
pip install representation-learning \
  --extra-index-url https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/
```

For development installation with training/evaluation tools, see the [Installation guide](docs/index.md#installation).

## Quick Start

```python
from representation_learning import list_models, load_model, describe_model

# List available models
models = list_models()
print(f"Available models: {list(models.keys())}")

# Get detailed information about a model
describe_model("beats_naturelm", verbose=True)

# Load a pre-trained model
model = load_model("beats_naturelm", device="cpu")

# Load for embedding extraction
model = load_model("beats_naturelm", return_features_only=True, device="cpu")
# Returns (batch, time_steps, 768) for BEATs

# Load with a probe for transfer learning
from representation_learning.api import build_probe_from_config
from representation_learning.configs import ProbeConfig

base = load_model("beats_naturelm", return_features_only=True, device="cpu")
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
    num_classes=10,
    device="cpu",
)
```

## Documentation and Examples

**Full documentation**: [docs/index.md](docs/index.md)

- [Getting Started](docs/index.md#getting-started) - Installation and quick start
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Architecture](docs/api_architecture.md) - Framework design
- [Training and Evaluation](docs/training_evaluation.md) - Training guides
- [Probe System](docs/probe_system.md) - Transfer learning with probes
- [Custom Models](docs/custom_model_registration.md) - Creating custom models

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
- Uses esp-data for dataset management
- Integrates with various pre-trained audio models
