# Representation Learning Framework

A comprehensive Python-based system for training, evaluating, and analyzing audio representation learning models with support for both supervised and self-supervised learning paradigms.

## 🚀 Quick Start

### Installation

**Method 1: Using uv (Recommended)**
```bash
# Clone the repository
git clone <repository-url>
cd representation-learning

# Install with uv
uv sync
```

**Method 2: Using pip**
```bash
# Install from source
pip install -e .

# Or install with private index for esp-data
pip install -e . --extra-index-url https://esp-pypi.com/simple/
```

**Method 3: Using setup.py**
```bash
python setup.py install
```

### Basic Usage

```python
from representation_learning import load_model, create_model, list_models

# List available models
models = list_models()
print(f"Available models: {list(models.keys())}")

# Load a pre-trained model
model = load_model("beats_naturelm", num_classes=10)

# Create a new model for training
model = create_model("efficientnet", num_classes=100, device="cuda")
```

## 📚 API Reference

### Model Loading Functions

The framework provides three main functions for working with models, each designed for specific use cases:

#### `load_model()` - Load Complete Models

**When to use:**
- ✅ Loading pre-trained models with weights
- ✅ Loading models from checkpoints
- ✅ Loading models for inference/evaluation
- ✅ When you need the full loading pipeline

**When NOT to use:**
- ❌ Creating new models for training from scratch
- ❌ When you don't need pre-trained weights
- ❌ Using custom model classes (use `build_model` for plugin architecture)

```python
from representation_learning import load_model

# Load with explicit num_classes (for new model)
model = load_model("efficientnet", num_classes=100)

# Load with custom checkpoint
model = load_model("efficientnet", checkpoint_path="gs://my-bucket/checkpoint.pt")

# Load with default checkpoint (if registered)
from representation_learning import register_checkpoint
register_checkpoint("beats_naturelm", "gs://my-bucket/beats_naturelm.pt")
model = load_model("beats_naturelm")  # Uses default checkpoint + extracts num_classes

# Load from config file
model = load_model("experiments/my_model.yml")

# Load with custom parameters
model = load_model("efficientnet", num_classes=50, device="cuda", efficientnet_variant="b1")
```

#### `create_model()` - Create New Models

**When to use:**
- ✅ Creating new models for training from scratch
- ✅ When you don't need pre-trained weights
- ✅ Using custom model classes (plugin architecture)
- ✅ Building models for fine-tuning

**When NOT to use:**
- ❌ Loading pre-trained models with weights
- ❌ Loading models from checkpoints
- ❌ Loading models for inference/evaluation

```python
from representation_learning import create_model

# Create new model for training
model = create_model("efficientnet", num_classes=100)

# Create custom model using plugin architecture
model = create_model("my_custom_model", num_classes=50)

# Create from config file
model = create_model("experiments/my_model.yml", num_classes=10)
```

#### `build_model()` - Plugin Architecture

**When to use:**
- ✅ Using the plugin architecture for new custom models
- ✅ When you have registered new model classes
- ✅ Building new models from ModelSpec objects

**When NOT to use:**
- ❌ Loading pre-trained models with weights (use `load_model`)
- ❌ Simple model creation (use `create_model`)

```python
from representation_learning import build_model, register_model_class
from representation_learning.models.base_model import ModelBase

# Register a custom model class
@register_model_class
class MyCustomModel(ModelBase):
    name = "my_custom_model"

    def __init__(self, device, num_classes, **kwargs):
        super().__init__(device=device)
        # Your model implementation
        pass

# Build using the plugin architecture
model = build_model("my_custom_model", device="cpu", num_classes=10)
```

### Model Registry Functions

#### Registry Management
```python
from representation_learning import (
    register_model, update_model, unregister_model,
    get_model, list_models, list_model_names, is_registered, describe_model
)

# Register a new model configuration
from representation_learning.configs import ModelSpec, AudioConfig
model_spec = ModelSpec(
    name="my_model",
    pretrained=False,
    device="cpu",
    audio_config=AudioConfig(sample_rate=16000)
)
register_model("my_model", model_spec)

# List available models
models = list_models()
print(f"Available models: {list(models.keys())}")

# Get model information
model_info = describe_model("efficientnet")
print(f"Model type: {model_info['_metadata']['model_type']}")

# Check if model is registered
if is_registered("efficientnet"):
    print("Model is available")
```

#### Model Class Management (Plugin Architecture)
```python
from representation_learning import (
    register_model_class, get_model_class, list_model_classes,
    is_model_class_registered, unregister_model_class
)

# Register a custom model class
@register_model_class
class MyModel(ModelBase):
    name = "my_model"
    # Implementation...

# List registered model classes
classes = list_model_classes()
print(f"Available model classes: {classes}")

# Get a specific model class
model_class = get_model_class("my_model")
```

#### Checkpoint Management
```python
from representation_learning import (
    register_checkpoint, get_checkpoint, unregister_checkpoint
)

# Register default checkpoint for a model
register_checkpoint("beats_naturelm", "gs://my-bucket/beats_naturelm.pt")

# Get checkpoint path
checkpoint = get_checkpoint("beats_naturelm")
print(f"Checkpoint: {checkpoint}")

# Unregister checkpoint
unregister_checkpoint("beats_naturelm")
```

## 🏗️ Architecture

### Core Components

1. **Model Registry** (`models/registry.py`)
   - Manages available model configurations
   - Thread-safe with lazy initialization
   - Supports dynamic model registration

2. **Model Factory** (`models/factory.py`)
   - Links ModelSpec configurations with model classes
   - Supports plugin architecture for custom models
   - Handles parameter extraction dynamically

3. **Model Loading** (`models/load.py`)
   - Provides unified interface for model loading
   - Supports checkpoint loading and weight extraction
   - Handles both registered and external models

4. **Base Model** (`models/base_model.py`)
   - Common functionality for all models
   - Hook management for embedding extraction
   - Audio processing capabilities

### Plugin Architecture

The framework supports a plugin architecture that allows users to register custom model classes without modifying the core library:

```python
from representation_learning.models.base_model import ModelBase
from representation_learning import register_model_class

@register_model_class
class MyCustomModel(ModelBase):
    name = "my_custom_model"

    def __init__(self, device: str, num_classes: int, **kwargs):
        super().__init__(device=device)
        # Your model implementation
        self.model = nn.Sequential(...)

    def forward(self, x, padding_mask=None):
        return self.model(x)

    def get_embedding_dim(self):
        return 512

# Now you can use it with any of the loading functions
model = create_model("my_custom_model", num_classes=10)
```

## 🎯 Supported Models

### Official Models

The framework includes support for various audio representation learning models:

- **EfficientNet**: Audio classification with different variants (b0, b1)
- **BEATs**: Self-supervised audio representation learning
- **EAT**: Audio transformer models (standard and HuggingFace versions)
- **AVES**: Audio-visual event detection
- **BirdMAE**: Bird-specific masked autoencoder

### Model Configuration

Models are configured using YAML files in the `configs/official_models/` directory:

```yaml
# Example: efficientnet_animalspeak.yml
model_spec:
  name: efficientnet
  pretrained: false
  device: cuda
  audio_config:
    sample_rate: 16000
    representation: mel_spectrogram
    n_mels: 128
  efficientnet_variant: b0

# Placeholder fields for RunConfig validation (not used for training)
training_params:
  train_epochs: 1
  lr: 0.001
  batch_size: 1
  optimizer: adamw
  weight_decay: 0.0

dataset_config:
  train_datasets:
    - dataset_name: placeholder
  val_datasets:
    - dataset_name: placeholder

output_dir: "./runs/placeholder"
loss_function: cross_entropy
```

## 🔧 Configuration

### ModelSpec Parameters

The `ModelSpec` class supports various parameters for different model types:

```python
from representation_learning.configs import ModelSpec, AudioConfig

# Basic configuration
model_spec = ModelSpec(
    name="efficientnet",
    pretrained=False,
    device="cuda",
    audio_config=AudioConfig(
        sample_rate=16000,
        representation="mel_spectrogram",
        n_mels=128
    )
)

# Model-specific parameters
model_spec = ModelSpec(
    name="beats",
    use_naturelm=True,
    fine_tuned=False,
    disable_layerdrop=False
)

# CLIP-specific parameters
model_spec = ModelSpec(
    name="clip",
    text_model_name="roberta-base",
    projection_dim=512,
    temperature=0.07
)

# EAT-specific parameters
model_spec = ModelSpec(
    name="eat_hf",
    model_id="worstchan/EAT-base_epoch30_pretrain",
    fairseq_weights_path="/path/to/weights.pt",
    eat_norm_mean=-4.268,
    eat_norm_std=4.569
)
```

### Audio Configuration

```python
from representation_learning.configs import AudioConfig

audio_config = AudioConfig(
    sample_rate=16000,
    n_fft=2048,
    hop_length=512,
    win_length=2048,
    window="hann",
    n_mels=128,
    representation="mel_spectrogram",  # or "spectrogram", "raw"
    normalize=True,
    target_length_seconds=10,
    window_selection="random",  # or "center"
    center=True
)
```

## 🚀 Training and Evaluation

### Training

```python
from representation_learning import load_model
from representation_learning.data import build_dataloaders

# Load model
model = load_model("efficientnet", num_classes=100)

# Prepare data
train_loader, val_loader = build_dataloaders(
    dataset_config=your_dataset_config,
    batch_size=32,
    num_workers=4
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Your training code
        pass
```

### Evaluation

```python
from representation_learning import load_model

# Load pre-trained model
model = load_model("beats_naturelm", checkpoint_path="path/to/checkpoint.pt")

# Set to evaluation mode
model.eval()

# Run evaluation
with torch.no_grad():
    for batch in eval_loader:
        outputs = model(batch["audio"])
        # Your evaluation code
```

## 📦 Package Structure

```
representation_learning/
├── __init__.py              # Main API exports
├── api/                     # Public API
│   ├── __init__.py
│   └── core.py             # Core API functions
├── configs/                 # Configuration schemas
│   ├── __init__.py
│   └── configs.py          # Pydantic models
├── data/                    # Data loading and processing
├── evaluation/              # Evaluation utilities
├── metrics/                 # Evaluation metrics
├── models/                  # Model implementations
│   ├── base_model.py       # Base model class
│   ├── factory.py          # Model factory
│   ├── get_model.py        # Original model factory
│   ├── load.py             # Model loading utilities
│   ├── registry.py         # Model registry
│   └── [model_files].py    # Individual model implementations
├── preprocessing/           # Audio preprocessing
├── training/                # Training utilities
└── utils/                   # Utility functions
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unittests/
uv run pytest tests/integration/
uv run pytest tests/consistency/

# Run with coverage
uv run pytest --cov=representation_learning
```

## 📝 Examples

### Custom Model Registration

```python
import torch
import torch.nn as nn
from representation_learning.models.base_model import ModelBase
from representation_learning import register_model_class, create_model

@register_model_class
class MyAudioCNN(ModelBase):
    name = "my_audio_cnn"

    def __init__(self, device: str, num_classes: int, **kwargs):
        super().__init__(device=device, **kwargs)

        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(64 * 25, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self.to(device)

    def forward(self, x, padding_mask=None):
        return self.model(x)

    def get_embedding_dim(self):
        return 128

# Use the custom model
model = create_model("my_audio_cnn", num_classes=10, device="cpu")
```

### Loading Pre-trained Models

```python
from representation_learning import load_model, register_checkpoint

# Register a checkpoint
register_checkpoint("beats_naturelm", "gs://my-bucket/beats_naturelm.pt")

# Load with default checkpoint
model = load_model("beats_naturelm")  # num_classes extracted from checkpoint

# Load with custom checkpoint
model = load_model("beats_naturelm", checkpoint_path="custom_checkpoint.pt")

# Load for different number of classes
model = load_model("beats_naturelm", num_classes=50)  # Override checkpoint classes
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of PyTorch
- Uses esp-data for dataset management
- Integrates with various pre-trained audio models
- Inspired by modern representation learning practices