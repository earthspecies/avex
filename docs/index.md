# Representation Learning Framework Documentation

Welcome to the Representation Learning Framework documentation. This framework provides an API for model loading and inference, and a Python-based system for training and evaluating bioacoustics representation learning models.

## Getting Started

### What is representation-learning?

The Representation Learning Framework is an API for model loading and inference, and a Python-based system for training and evaluating bioacoustics representation learning models. It provides:

- **Unified API** for loading and using pre-trained audio models
- **Model Registry** for managing model configurations
- **Plugin Architecture** for custom model registration
- **Training and Evaluation** tools for bioacoustics tasks
- **Embedding Extraction** capabilities for downstream tasks

### Quick Start

#### Basic Usage

```python
from representation_learning import list_models, load_model, describe_model

# List available models (prints table and returns dict)
models = list_models()  # Prints table + returns dict with detailed info
print(f"Available models: {list(models.keys())}")

# Get detailed information about a model
describe_model("beats_naturelm", verbose=True)
# Shows: model type, whether it has a trained classifier, number of classes, usage examples

# Load a pre-trained model with checkpoint (num_classes extracted automatically)
model = load_model("sl_beats_animalspeak", device="cpu")

# For a new task, load a backbone and attach a probe head (classifier)
base = load_model("beats_naturelm", return_features_only=True, device="cpu")

# Load for embedding extraction (returns unpooled features)
model = load_model("beats_naturelm", return_features_only=True, device="cpu")
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
from representation_learning import load_model
from representation_learning.models.probes import build_probe_from_config
from representation_learning.configs import ProbeConfig

# Load backbone for feature extraction
base = load_model("beats_naturelm", return_features_only=True, device="cpu")

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
- **Developer**: you want to clone the repo, modify code, or run the full training/evaluation stack.

### 1. API Usage

For users who want to install the package and use it as a library (for example to load models and run inference).

#### 1.1 Prerequisites

- Python 3.10, 3.11, or 3.12

#### 1.2 Install with uv (recommended)

This assumes you are using `uv` to manage your project or environment.

1. Add the following to your `pyproject.toml` list of dependencies (either create one or edit the existing one):

```toml

dependencies = [
  "representation-learning",
]
```

**Note:** If you plan to install `representation-learning[dev]` (see section 2), you need to include `esp-data` and `esp-sweep` in `[tool.uv.sources]` as shown above, since they are dependencies of the `dev` extras and also come from the esp-pypi index.

2. Install the package (API dependencies only):

```bash
# Option A: Add and install in one step
uv add representation-learning

# Option B: If you've already added it to [project.dependencies] in pyproject.toml
uv sync
```

#### 1.3 Install with pip

If you prefer plain `pip`:

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the package from the ESP index:

```bash
pip install representation-learning
```

#### 2. API + full dependencies (training / evaluation)

If you want to use additional functionality such as `run_train.py`, `run_evaluate.py`, or other advanced workflows, install the `dev` extras:

#### 2.1 Prerequisites

- ESP GCP authentication:

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login
```

#### 2.2 Install with uv (recommended)

This assumes you are using `uv` to manage your project or environment.

1. Install keyring with the Google Artifact Registry plugin (once per machine):

```bash
uv tool install keyring --with keyrings.google-artifactregistry-auth
```

2. Configure `uv` to use the internal ESP PyPI index. Add the following to your `pyproject.toml` (either create one or edit the existing one):

```toml
[[tool.uv.index]]
name = "esp-pypi"
url = "https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/"
explicit = true

[tool.uv.sources]
representation-learning = { index = "esp-pypi" }
# Optional: only needed if you plan to install the dev extras (representation-learning[dev])
esp-data = { index = "esp-pypi" }
esp-sweep = { index = "esp-pypi" }

[tool.uv]
keyring-provider = "subprocess"
```

3. Install the package (with [dev] dependencies):
```bash
# With uv (in a project configured for esp-pypi as above)

# Option A: Add and install in one step
uv add "representation-learning[dev]"

# Option B: If you've already added it to pyproject.toml
uv sync

# With pip
pip install "representation-learning[dev]" \
  --extra-index-url https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/
```

This pulls in additional dependencies, including for example:

- `pytorch-lightning` – training (for ATST)
- `mlflow` – experiment tracking
- `wandb` – Weights & Biases integration
- `esp-sweep` – hyperparameter sweeping
- `esp-data` – dataset management
- `gradio` – interactive demos
- `gradio-leaderboard` – leaderboard visualization

### 2. Development Usage

For contributors or power users who clone the repository and want the full development and runtime stack locally.

#### 2.1 Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- GCP authentication:

```bash
gcloud auth login
gcloud auth application-default login
```

#### 2.2 Clone the repository

```bash
git clone <repository-url>
cd representation-learning
```

#### 2.3 Install with uv (recommended for development)

```bash
# 1. Install keyring with Google Artifact Registry plugin
uv tool install keyring --with keyrings.google-artifactregistry-auth

# 2. Install the project with all dev/runtime dependencies
uv sync --group project-dev
```

This will install:

- Base API dependencies
- Training/evaluation runtime dependencies (for example `pytorch-lightning`, `mlflow`, `wandb`, `esp-data`, etc.)
- Development tools (`pytest`, `ruff`, `pre-commit`, etc.)
- Optional GPU-related packages (for example `bitsandbytes`, when supported)

The `project-dev` dependency group is used by CI and is intended to match the full development environment.

#### 2.4 Install with pip (alternative for development)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install in editable mode with dev extras
pip install -e ".[dev]" \
  --extra-index-url https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/
```

Notes for development:

- Editable install (`-e`) means changes in the repo are picked up immediately without reinstalling.
- The `[dev]` extra mirrors the runtime dependencies used by `uv`'s `project-dev` group.
- Use this setup if you plan to:
  - Run tests (`pytest`)
  - Run training/evaluation scripts
  - Contribute code via pull requests

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
