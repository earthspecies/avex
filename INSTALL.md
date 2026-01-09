# Installation Guide

This guide explains how to install the `representation-learning` package.

The installation process depends on how you plan to use this package:

- **API user**: you just want to load models and run inference.
- **Developer**: you want to clone the repo, modify code, or run the full training/evaluation stack.

## 1. API Usage

For users who want to install the package and use it as a library (for example to load models and run inference).

### 1.1 Prerequisites

- Python 3.10, 3.11, or 3.12
- ESP GCP authentication:

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login
```

### 1.2 Install with uv (recommended)

This assumes you are using `uv` to manage your project or environment.

1. Install keyring with the Google Artifact Registry plugin (once per machine):

```bash
uv tool install keyring --with keyrings.google-artifactregistry-auth
```

**Note for Slurm users**: This step is NOT required for Slurm jobs. All nodes on the cluster already have this package installed.

2. Create and activate a uv-managed virtual environment (if you do not already have one):

```bash
uv venv
source .venv/bin/activate
```

3. Configure `uv` to use the internal ESP PyPI index. Add the following to your `pyproject.toml` (either create one or edit the existing one):

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

**Note:** If you plan to install `representation-learning[dev]` (see section 1.4), you need to include `esp-data` and `esp-sweep` in `[tool.uv.sources]` as shown above, since they are dependencies of the `dev` extras and also come from the esp-pypi index.

4. Install the package (API dependencies only):

```bash
# Option A: Add and install in one step
uv add representation-learning

# Option B: If you've already added it to [project.dependencies] in pyproject.toml
uv sync
```

### 1.3 Install with pip

If you prefer plain `pip`:

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the package from the ESP index:

```bash
pip install representation-learning \
  --extra-index-url https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/
```

### 1.4 API + full dependencies (training / evaluation)

If you want to use additional functionality such as `run_train.py`, `run_evaluate.py`, or other advanced workflows, install the `dev` extras:

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

## 2. Development Usage

For contributors or power users who clone the repository and want the full development and runtime stack locally.

### 2.1 Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- GCP authentication:

```bash
gcloud auth login
gcloud auth application-default login
```

### 2.2 Clone the repository

```bash
git clone <repository-url>
cd representation-learning
```

### 2.3 Install with uv (recommended for development)

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

### 2.4 Install with pip (alternative for development)

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

## Verification

After installation, verify that everything works:

```python
# Test the API
from avex import load_model, list_models

# List available models
models = list_models()
print(f"Available models: {list(models.keys())}")

# Load a model
model = load_model("efficientnet_animalspeak", num_classes=10, device="cpu")
print(f"Model loaded: {type(model).__name__}")
```

```bash
# Test CLI commands
list-models
```

## Troubleshooting

### esp-data Not Found

If you get an error about `esp-data` not being found:

1. Make sure you've configured the private index in your `pyproject.toml` (see section 1.2)
2. Check that you have access to the Earth Species private PyPI repository
3. Verify your authentication token is valid
4. Make sure you've installed keyring (see section 1.2, step 1)

### Permission Errors

If you get permission errors during installation, ensure you have the correct permissions to access the private repository.

### CUDA Issues

If you encounter CUDA-related issues, you can install CPU-only PyTorch:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Dependencies

The package requires several dependencies including:

- **Core ML**: PyTorch, Transformers, Timm
- **Audio**: Librosa, SoundFile, Resampy
- **Data**: Pandas, NumPy, H5Py
- **Cloud**: Google Cloud Storage, CloudPathLib
- **Private**: esp-data, esp-sweep (from Earth Species private PyPI)

See `pyproject.toml` for the complete list of dependencies.
