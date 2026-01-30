# Representation Learning Framework Documentation

Welcome to the Representation Learning Framework documentation. This framework provides an API for model loading and inference, and a Python-based system for training and evaluating bioacoustics representation learning models.

## Getting Started

### What is avex?

The Representation Learning Framework is an API for model loading and inference, and a Python-based system for training and evaluating bioacoustics representation learning models. It provides:

- **Unified API** for loading and using pre-trained audio models
- **Model Registry** for managing model configurations
- **Plugin Architecture** for custom model registration
- **Training and Evaluation** tools for bioacoustics tasks
- **Embedding Extraction** capabilities for downstream tasks

### Quick Start

#### Basic Usage

```python
from avex import list_models, load_model, describe_model

# List available models (prints table and returns dict)
models = list_models()  # Prints table + returns dict with detailed info
print(f"Available models: {list(models.keys())}")

# Get detailed information about a model
describe_model("esp_aves2_naturelm_audio_v1_beats", verbose=True)
# Shows: model type, whether it has a trained classifier, number of classes, usage examples

# Load a pre-trained model with checkpoint (num_classes extracted automatically)
model = load_model("esp_aves2_sl_beats_all", device="cpu")

# For a new task, load a backbone and attach a probe head (classifier)
base = load_model("esp_aves2_naturelm_audio_v1_beats", return_features_only=True, device="cpu")

# Load for embedding extraction (returns unpooled features)
model = load_model("esp_aves2_naturelm_audio_v1_beats", return_features_only=True, device="cpu")
# Returns (batch, time_steps, 768) for BEATs instead of classification logits
```

#### Probes (Heads on Top of Backbones)

Probes are task-specific heads attached to pretrained backbones for transfer learning:

```
┌─────────────────────────────────────────────────────────┐
│                    Audio Input                          │
│              (batch, time_steps)                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Pretrained Backbone │
         │   (e.g., BEATs)       │
         │                       │
         │  ┌─────────────────┐  │
         │  │  Layer 1        │  │
         │  └────────┬────────┘  │
         │           │           │
         │  ┌────────▼────────┐  │
         │  │  Layer 2        │  │
         │  └────────┬────────┘  │
         │           │           │
         │  ┌────────▼────────┐  │
         │  │  ...            │  │
         │  └────────┬────────┘  │
         │           │           │
         │  ┌────────▼────────┐  │
         │  │  Last Layer     │◄─┼── target_layers=["last_layer"]
         │  └────────┬────────┘  │
         └───────────┼───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Embeddings          │
         │   (batch, dim)        │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Probe Head          │
         │   (linear/MLP/etc.)   │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Task Predictions    │
         │   (batch, num_classes)│
         └───────────────────────┘
```

**Key Concepts:**
- **Backbone**: Pretrained model (frozen or fine-tunable)
- **Probe**: Task-specific head (trained for your task)
- **Target Layers**: Which backbone layers to extract features from
- **Online Training**: Backbone + probe trained together
- **Offline Training**: Embeddings pre-computed, probe trained separately

```python
from avex import load_model
from avex.models.probes import build_probe_from_config
from avex.configs import ProbeConfig

# Load backbone for feature extraction
base = load_model("esp_aves2_naturelm_audio_v1_beats", return_features_only=True, device="cpu")

# Define a simple linear probe on the backbone features
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

> **Note**: Each model expects a specific sample rate (e.g., 16 kHz for BEATs, 32 kHz for Perch). Use `describe_model()` to check, and resample with `librosa.resample()` if needed. See [Audio Requirements](configuration.md#audio-requirements) for details.

For more examples, see the `examples/` directory:
- `00_quick_start.py` - Basic model loading and testing
- `01_basic_model_loading.py` - Loading models with different configurations
- `02_checkpoint_loading.py` - Working with checkpoints and class mappings
- `03_custom_model_registration.py` - Creating and registering custom models and ModelSpecs
- `04_training_and_evaluation.py` - Full training loop and evaluation examples
- `05_embedding_extraction.py` - Feature extraction with `return_features_only=True` (unpooled features)
- `06_classifier_head_loading.py` - Understanding classifier head behavior

## Installation

The installation process depends on how you plan to use this package:

- **API user**: you just want to load models and run inference.
- **Developer**: you want to clone the repo, modify code, or contribute.

### 1. API Usage

For users who want to install the package and use it as a library (for example to load models and run inference).

#### Prerequisites

- Python 3.10, 3.11, or 3.12

#### Install with pip

```bash
pip install avex
```

#### Install with uv

```bash
# Option A: Add and install in one step
uv add avex

# Option B: If you've already added it to [project.dependencies] in pyproject.toml
uv sync
```

### 2. Training Setup (ESP Internal)

Training the original supervised learning models requires `esp-data` for dataset management. This package is currently only available to Earth Species Project team members.

#### Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- GCP authentication:

```bash
gcloud auth login
gcloud auth application-default login
```

#### Install with uv (recommended)

1. Install keyring with the Google Artifact Registry plugin (once per machine):

```bash
uv tool install keyring --with keyrings.google-artifactregistry-auth
```

2. Configure `uv` to use the internal ESP PyPI index. Add the following to your `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "esp-pypi"
url = "https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/"
explicit = true

[tool.uv.sources]
avex = { index = "esp-pypi" }
esp-data = { index = "esp-pypi" }
esp-sweep = { index = "esp-pypi" }

[tool.uv]
keyring-provider = "subprocess"
```

3. Install with training dependencies:

```bash
uv add "avex[dev]"
```

#### Install with pip

```bash
pip install "avex[dev]" \
  --extra-index-url https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/
```

This installs additional dependencies for training:
- `esp-data` – dataset management
- `esp-sweep` – hyperparameter sweeping
- `pytorch-lightning`, `mlflow`, `wandb` – training infrastructure

For contributing to the codebase, see [CONTRIBUTING.md](../CONTRIBUTING.md)

## Core Documentation

- **[API Reference](api_reference.md)** - Complete API documentation for model loading, registry, and management functions
- **[Architecture](api_architecture.md)** - Framework architecture, core components, and plugin system
- **[Supported Models](supported_models.md)** - List of supported models and their configurations
- **[Configuration](configuration.md)** - ModelSpec parameters, audio requirements, and configuration options

## Usage Guides

- **[Training and Evaluation](training_evaluation.md)** - Guide to training and evaluating models
- **[Embedding Extraction](embedding_extraction.md)** - Working with feature representations and embeddings
- **[Examples](examples.md)** - Comprehensive examples and use cases

## Advanced Topics

- **[Probe System](probe_system.md)** - Understanding and using probes for transfer learning
- **[API Probes](api_probes.md)** - API reference for probe-related functionality
- **[Custom Model Registration](custom_model_registration.md)** - Guide on registering custom model classes and loading pre-trained models

## Contributing

- **[Contributing Guide](../CONTRIBUTING.md)** - Instructions for contributing to the project, including development setup, testing, and pull request process
