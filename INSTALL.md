# Installation Guide

This guide explains how to install the `representation-learning` package.

## Prerequisites

- Python 3.11 or 3.12
- pip or uv package manager

## Installation Methods

### Method 1: Using uv (Recommended)

The easiest and most reliable way to install the package:

```bash
# Clone the repository
git clone https://github.com/earthspecies/representation-learning.git
cd representation-learning

# Install with uv (handles private index automatically)
uv sync
```

### Method 2: Using the Setup Script

Alternative installation using the provided setup script:

```bash
# Clone the repository
git clone https://github.com/earthspecies/representation-learning.git
cd representation-learning

# Install with private dependencies
python setup.py
```

### Method 3: Using pip with Private Index

```bash
# Install with the private esp-data index
pip install -e . --extra-index-url https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/
```

### Method 4: From Built Wheel

```bash
# Build the package
python -m build

# Install from wheel
python setup.py --wheel
```

## Verification

After installation, verify that everything works:

```python
# Test the API
from representation_learning import load_model, list_models

# List available models
models = list_models()
print(f"Available models: {list(models.keys())}")

# Load a new model with explicit num_classes
model = load_model("efficientnet", num_classes=100)
print(f"Model loaded: {type(model).__name__}")

# Load from checkpoint (num_classes extracted automatically)
model = load_model("efficientnet", checkpoint_path="path/to/checkpoint.pt")
print(f"Model loaded from checkpoint: {type(model).__name__}")

# Load with default checkpoint (if model has one registered)
from representation_learning import register_checkpoint
register_checkpoint("beats_naturelm", "gs://my-bucket/beats_naturelm.pt")
model = load_model("beats_naturelm")  # Uses default checkpoint + extracts num_classes
print(f"Model loaded with default checkpoint: {type(model).__name__}")
```

```bash
# Test CLI commands
list-models
list-models --detailed
```

## Troubleshooting

### esp-data Not Found

If you get an error about `esp-data` not being found:

1. Make sure you're using the private index URL
2. Check that you have access to the Earth Species private PyPI repository
3. Verify your authentication token is valid

### Permission Errors

If you get permission errors during installation:

```bash
# Install in user space
pip install -e . --user --extra-index-url https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/
```

### CUDA Issues

If you encounter CUDA-related issues:

```bash
# Install CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Development Installation

For development work:

```bash
# Clone and install in editable mode
git clone https://github.com/earthspecies/representation-learning.git
cd representation-learning
uv sync
```

## Dependencies

The package requires several dependencies including:

- **Core ML**: PyTorch, Transformers, Timm
- **Audio**: Librosa, SoundFile, Resampy
- **Data**: Pandas, NumPy, H5Py
- **Cloud**: Google Cloud Storage, CloudPathLib
- **Private**: esp-data, esp-sweep (from Earth Species private PyPI)

See `pyproject.toml` for the complete list of dependencies.
