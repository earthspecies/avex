# Installation Guide

This guide explains how to install the `representation-learning` package.

## Prerequisites

- Python 3.10, 3.11, or 3.12
- `uv` package manager (recommended) or `pip`

## Installation

`representation-learning` is currently a private package, hosted on ESP's internal Python package repository. Because it isn't available on the public PyPI index, you'll need to configure your project to use ESP's private package index in order to install and update it.

### 1. Install keyring (one-time setup)

To authenticate and interact with Python repositories hosted on Artifact Registry, you'll need to install the keyring library system-wide (not inside a virtual environment), along with the Google Artifact Registry backend. This step is required only once per system, typically when setting up your VM or laptop (not needed on Slurm compute nodes):

```bash
uv tool install keyring --with keyrings.google-artifactregistry-auth
```

**Slurm**

This step is NOT required for Slurm jobs. All nodes on the cluster already have this package installed.

> **Info**
>
> You only need to do this step once on your system.

> **Tip**
>
> `uv tool` allows you to install Python packages that provide command-line interfaces for system-wide use. The dependencies are installed in an isolated virtual environment, separate from your current project.

### 2. Choose your installation scenario

There are three scenarios for installing `representation-learning`:

#### Scenario 1: Just want to use the package

You just want to use `representation-learning` package and don't care about its source code and implementation. In that case, the installation is very similar to the esp-data guide.

**Configure your project to use the private index**

Add the following to your `pyproject.toml` to configure your project to use the private package index:

```toml
[[tool.uv.index]]
name = "esp-pypi"
url = "https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/"
explicit = true

[tool.uv.sources]
representation-learning = { index = "esp-pypi" }

[tool.uv]
keyring-provider = "subprocess"
```

**Add representation-learning as a dependency**

You can now add `representation-learning` to your project by running:

```bash
uv add representation-learning
```

Alternatively, you can manually update the dependencies section of your `pyproject.toml` and then run:

```bash
uv sync
```

#### Scenario 2: Use as dependency but want to hack/patch the code

You want to use `representation-learning` as a dependency but want to hack/patch the code. In that case, clone this repo and add it as an editable dependency:

```bash
# Clone the repository
git clone https://github.com/earthspecies/representation-learning.git
cd representation-learning

# In your project, add as editable dependency
uv add --editable "/path/to/representation-learning"
```

Or manually add to your `pyproject.toml`:

```toml
[tool.uv.sources]
representation-learning = { path = "/path/to/representation-learning", editable = true }
```

Then run:

```bash
uv sync
```

#### Scenario 3: You're a developer of this package

You're a developer of this package. In that case, you just clone and do `uv sync`:

```bash
# Clone the repository
git clone https://github.com/earthspecies/representation-learning.git
cd representation-learning

# Install dependencies (handles private index automatically)
uv sync
```

This will install all dependencies including `esp-data` and `esp-sweep` from the private index, as configured in the repository's `pyproject.toml`.

## Verification

After installation, verify that everything works:

```python
# Test the API
from representation_learning import load_model, list_models

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

1. Make sure you've configured the private index in your `pyproject.toml` (see Scenario 1)
2. Check that you have access to the Earth Species private PyPI repository
3. Verify your authentication token is valid
4. Make sure you've installed keyring (see step 1)

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
