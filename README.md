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
from representation_learning import list_models, load_model, describe_model

# List available models
models = list_models()
print(f"Available models: {list(models.keys())}")

# Get detailed information about a model
describe_model("beats_naturelm", verbose=True)

# Load a pre-trained model with checkpoint (num_classes extracted automatically)
model = load_model("sl_beats_animalspeak", num_classes=None, device="cpu")

# Load a model for a new task (creates new classifier)
model = load_model("beats_naturelm", num_classes=10, device="cpu")

# Load for embedding extraction (no classifier)
model = load_model("beats_naturelm", num_classes=None, return_features_only=True, device="cpu")
```

For more examples, see the `examples/` directory:
- `00_quick_start.py` - Basic model loading and testing
- `01_basic_model_loading.py` - Loading models with different configurations
- `02_checkpoint_loading.py` - Working with checkpoints and class mappings
- `03_custom_model_registration.py` - Creating and registering custom models
- `06_embedding_extraction.py` - Feature extraction mode
- `07_classifier_head_loading.py` - Understanding classifier head behavior

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

# Load with explicit num_classes (for new model with new classifier)
model = load_model("efficientnet", num_classes=100)

# Load with custom checkpoint
model = load_model("efficientnet", checkpoint_path="gs://my-bucket/checkpoint.pt")

# Load with default checkpoint (from YAML config)
# Checkpoint paths are defined in api/configs/official_models/*.yml files
# When num_classes=None, it's automatically extracted from the checkpoint
model = load_model("efficientnet_animalspeak")  # Uses default checkpoint from YAML + extracts num_classes

# Load for embedding extraction (no classifier head)
# When num_classes=None and no checkpoint, builds model for embedding extraction
model = load_model("beats", num_classes=None)  # Returns embeddings, not logits

# Load from config file
model = load_model("experiments/my_model.yml")

# Load with custom parameters
model = load_model("efficientnet", num_classes=50, device="cuda", efficientnet_variant="b1")
```

**Important: `num_classes` and `pretrained` Parameter Behavior**

The `num_classes` parameter has different behaviors depending on the context:

1. **`num_classes=None` with checkpoint**:
   - Extracts `num_classes` from the checkpoint automatically
   - Loads the classifier weights from the checkpoint (preserves trained classifier)
   - Example: `load_model("efficientnet_animalspeak")` - extracts classes from checkpoint

2. **`num_classes=None` without checkpoint**:
   - If the model supports `return_features_only=True`, builds the model for embedding extraction
   - No classifier head is added (returns embeddings instead of logits)
   - Example: `load_model("beats", num_classes=None)` - for embedding extraction

3. **`num_classes` explicitly provided**:
   - Creates a new classifier head with the specified number of classes
   - If a checkpoint is provided, the classifier weights are NOT loaded (randomly initialized)
   - Example: `load_model("efficientnet", num_classes=50)` - new classifier with 50 classes

**`pretrained=True` Behavior**

When `pretrained=True` and no `checkpoint_path` is provided:

- **BEATs**: Always loads pretrained weights from hardcoded paths (ImageNet-pretrained or SSL-pretrained) regardless of `pretrained` flag
- **EfficientNet**: Uses `pretrained=True` to load ImageNet-pretrained weights via torchvision
- **EAT-HF**: Loads pretrained weights from HuggingFace when `pretrained=True`
- **Other models**: Each model type has its own pretrained weight loading mechanism

**Important**: If a `checkpoint_path` is provided (either explicitly or from YAML), `pretrained` is automatically set to `False` to avoid conflicts. The checkpoint weights take priority over pretrained weights.

```python
# Load with pretrained=True (no checkpoint) - uses model's own pretrained weights
model = load_model("beats", pretrained=True, num_classes=None)  # BEATs loads SSL weights

# Load with checkpoint - pretrained is automatically False
model = load_model("efficientnet_animalspeak")  # Uses checkpoint, pretrained=False
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
    register_model, get_model_spec, list_models, describe_model
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
model_spec = get_model_spec("efficientnet")
if model_spec is not None:
    print("Model is available")
else:
    print("Model is not available")
```

#### Model Class Management (Plugin Architecture)
```python
from representation_learning import (
    register_model_class, get_model_class, list_model_classes
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
from representation_learning import get_checkpoint_path

# Get default checkpoint path from YAML config
# Checkpoint paths are defined in api/configs/official_models/*.yml files
checkpoint = get_checkpoint_path("efficientnet_animalspeak")
print(f"Default checkpoint: {checkpoint}")

# Override default checkpoint by passing checkpoint_path parameter
from representation_learning import load_model
model = load_model("efficientnet_animalspeak", checkpoint_path="gs://my-custom-checkpoint.pt")
```

#### Class Mapping Management
```python
from representation_learning import load_class_mapping

# Load class mappings for a model
# Class mappings define the relationship between class labels and indices
class_mapping = load_class_mapping("sl_beats_animalspeak")
if class_mapping:
    label_to_index = class_mapping["label_to_index"]
    index_to_label = class_mapping["index_to_label"]
    print(f"Loaded {len(label_to_index)} classes")
    print(f"Example: {label_to_index['dog']}")  # Get index for 'dog'
    print(f"Example: {index_to_label[0]}")  # Get label for index 0
```

## 🏗️ Architecture

### Core Components

1. **Model Registry** (`models/utils/registry.py`)
   - Manages available model configurations
   - Thread-safe with lazy initialization
   - Supports dynamic model registration

2. **Model Factory** (`models/utils/factory.py`)
   - Links ModelSpec configurations with model classes
   - Supports plugin architecture for custom models
   - Handles parameter extraction dynamically

3. **Model Loading** (`models/utils/load.py`)
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

Models are configured using YAML files in the `api/configs/official_models/` directory:

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
import torch
import torch.nn as nn
import torch.optim as optim
from representation_learning import create_model

# Create a model for training
model = create_model("efficientnet", num_classes=100, device="cpu")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_audio, batch_labels in train_loader:
        # Forward pass
        outputs = model(batch_audio, padding_mask=None)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save checkpoint
torch.save(model.state_dict(), "checkpoints/my_model.pt")
```

For complete training examples with data loading and evaluation, see `examples/05_training_and_evaluation.py`.

### Evaluation

```python
import torch
from representation_learning import load_model

# Load pre-trained model with checkpoint
model = load_model("sl_beats_animalspeak", num_classes=None, device="cpu")

# Set to evaluation mode
model.eval()

# Run evaluation
correct = 0
total = 0
with torch.no_grad():
    for batch_audio, batch_labels in eval_loader:
        outputs = model(batch_audio, padding_mask=None)
        _, predicted = torch.max(outputs, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")
```

## 📦 Package Structure

```
representation_learning/
├── __init__.py              # Main API exports and version
├── api/                     # Public API layer
│   ├── configs/            # Official model configurations
│   │   └── official_models/  # YAML configs for official models
│   └── list_models.py      # CLI utility for listing models
├── cli.py                   # Command-line interface
├── configs.py               # Pydantic configuration models
├── data/                    # Data loading and processing
├── evaluation/              # Evaluation utilities
├── metrics/                 # Evaluation metrics
├── models/                  # Model implementations
│   ├── utils/              # Model utilities (factory, load, registry)
│   ├── probes/             # Probe implementations
│   ├── beats/              # BEATs model components
│   ├── eat/                # EAT model components
│   └── atst_frame/         # ATST-Frame model components
├── preprocessing/           # Audio preprocessing
├── training/                # Training utilities
├── utils/                   # Utility functions
├── run_train.py            # Training entry point
└── run_evaluate.py         # Evaluation entry point
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

The `examples/` directory contains comprehensive examples demonstrating various usage patterns:

| Example | Description |
|---------|-------------|
| `00_quick_start.py` | Basic model loading and testing |
| `01_basic_model_loading.py` | Loading pre-trained models with checkpoints and class mappings |
| `02_checkpoint_loading.py` | Working with default and custom checkpoints from YAML configs |
| `03_custom_model_registration.py` | Creating and registering custom model classes |
| `04_model_registry_management.py` | Managing model configurations and registrations |
| `05_training_and_evaluation.py` | Full training loop and evaluation examples |
| `06_embedding_extraction.py` | Feature extraction mode with `return_features_only=True` |
| `07_classifier_head_loading.py` | Understanding classifier head behavior with different `num_classes` settings |
| `colab_sl_beats_demo.ipynb` | Google Colab demo for the sl-beats model |

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

**Checkpoint Path Management**

Checkpoint paths are now managed directly in YAML configuration files (`api/configs/official_models/*.yml`). The framework reads checkpoint paths from YAML when needed, eliminating the need for a separate checkpoint registry.

```python
from representation_learning import load_model, get_checkpoint_path

# Checkpoint paths are defined in YAML files (api/configs/official_models/*.yml)
# Get default checkpoint path (read from YAML)
# Returns None if the model doesn't have checkpoint_path
checkpoint = get_checkpoint_path("efficientnet_animalspeak")
print(f"Default checkpoint: {checkpoint}")

# Load with default checkpoint (from YAML)
# num_classes=None automatically extracts num_classes from checkpoint
# and preserves the classifier weights from the checkpoint
model = load_model("efficientnet_animalspeak")  # Uses YAML checkpoint + extracts num_classes

# Load with custom checkpoint (overrides YAML default)
# Priority: user-provided checkpoint_path > YAML default > no checkpoint
model = load_model("efficientnet_animalspeak", checkpoint_path="gs://my-custom-checkpoint.pt")

# Load for different number of classes
# When num_classes is explicitly provided, a new classifier is created
# (checkpoint classifier weights are NOT loaded - randomly initialized)
model = load_model("efficientnet_animalspeak", num_classes=50)  # New classifier with 50 classes
```

**Checkpoint Path Priority**

When loading a model, checkpoint paths are resolved in this order:
1. **User-provided `checkpoint_path` parameter** (highest priority)
2. **Default checkpoint from YAML file** (if `num_classes=None`)
3. **No checkpoint** (for embedding extraction or new models)

**Classifier Head Behavior**

- **`num_classes=None` with checkpoint**: Extracts `num_classes` from checkpoint and preserves classifier weights
- **`num_classes=None` without checkpoint**: Builds model for embedding extraction (if supported)
- **`num_classes` explicitly set**: Creates new classifier head (checkpoint classifier weights are ignored)

**`pretrained=True` Without Checkpoint**

When `pretrained=True` and no `checkpoint_path` is set:
- The model uses its own pretrained weight loading mechanism (varies by model type)
- BEATs: Loads from hardcoded SSL/ImageNet paths
- EfficientNet: Loads ImageNet weights via torchvision
- EAT-HF: Loads from HuggingFace
- **Note**: If a `checkpoint_path` is found (from YAML or user-provided), `pretrained` is automatically set to `False` to prioritize checkpoint weights

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