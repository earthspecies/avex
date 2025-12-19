# Representation Learning Framework

An API for model loading and inference, and a Python-based system for training and evaluating bioacoustics representation learning models.

## 🚀 Quick Start

### Installation

The installation process depends on how you plan to use this package:

- **API user**: you just want to load models and run inference.
- **Developer**: you want to clone the repo, modify code, or run the full training/evaluation stack.

### 1. API Usage

For users who want to install the package and use it as a library (for example to load models and run inference).

#### 1.1 Prerequisites

- Python 3.10, 3.11, or 3.12
- ESP GCP authentication:

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login
```

#### 1.2 Install with uv (recommended)

This assumes you are using `uv` to manage your project or environment.

1. Install keyring with the Google Artifact Registry plugin (once per machine):

```bash
uv tool install keyring --with keyrings.google-artifactregistry-auth
```

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

#### 1.3 Install with pip

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

#### 1.4 API + full dependencies (training / evaluation)

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
- The `[dev]` extra mirrors the runtime dependencies used by `uv`’s `project-dev` group.
- Use this setup if you plan to:
  - Run tests (`pytest`)
  - Run training/evaluation scripts
  - Contribute code via pull requests

### Basic Usage

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

# Load a model for a new task (creates new classifier)
model = load_model("beats_naturelm", num_classes=10, device="cpu")

# Load for embedding extraction (returns unpooled features)
model = load_model("beats_naturelm", return_features_only=True, device="cpu")
# Returns (batch, time_steps, 768) for BEATs instead of classification logits
```

#### Probes (Heads on Top of Backbones)

```python
from representation_learning import load_model
from representation_learning.api import build_probe_from_config_online
from representation_learning.configs import ProbeConfig

# Load backbone for feature extraction
base = load_model("beats_naturelm", return_features_only=True, device="cpu")

# Define a simple linear probe on the backbone features
probe_config = ProbeConfig(
    probe_type="linear",
    target_layers=["backbone"],
    aggregation="mean",
    freeze_backbone=True,
    online_training=True,
)

probe = build_probe_from_config_online(
    probe_config=probe_config,
    base_model=base,
    num_classes=10,
    device="cpu",
)
```

> **Note**: Each model expects a specific sample rate (e.g., 16 kHz for BEATs, 32 kHz for Perch). Use `describe_model()` to check, and resample with `librosa.resample()` if needed. See [Audio Requirements](#audio-requirements) for details.

For more examples, see the `examples/` directory:
- `00_quick_start.py` - Basic model loading and testing
- `01_basic_model_loading.py` - Loading models with different configurations
- `02_checkpoint_loading.py` - Working with checkpoints and class mappings
- `03_custom_model_registration.py` - Creating and registering custom models
- `04_model_registry_management.py` - Managing model configurations and registrations
- `05_training_and_evaluation.py` - Full training loop and evaluation examples
- `06_embedding_extraction.py` - Feature extraction with `return_features_only=True` (unpooled features)
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
model = load_model("beats")  # Returns embeddings, not logits

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

2. **`num_classes=None` without checkpoint** (default behavior):
   - If the model supports `return_features_only=True`, builds the model for embedding extraction
   - No classifier head is added (returns embeddings instead of logits)
   - Example: `load_model("beats")` - for embedding extraction

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
model = load_model("beats", pretrained=True)  # BEATs loads SSL weights

# Load with checkpoint - pretrained is automatically False
model = load_model("efficientnet_animalspeak")  # Uses checkpoint, pretrained=False
```

#### Training New Models

**Recommended pattern:**
- Define a custom model class (subclassing `ModelBase`) with its own classifier head, or
- Build a backbone via `build_model` / `build_model_from_spec` and attach a probe head with `build_probe_from_config_online` or `build_probe_from_config_offline`.

#### `build_model()` - Plugin Architecture

**When to use:**
- ✅ Using the plugin architecture for new custom models
- ✅ When you have registered new model classes
- ✅ Building new models from ModelSpec objects

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

# List available models (prints table and returns dict)
models = list_models()
# Prints formatted table:
# ====================================================================================================
# Model Name                          Description                              Trained Classifier
# ====================================================================================================
# beats_naturelm                      beats (pretrained backbone) - NatureLM   ❌ No
# sl_beats_animalspeak                beats (fine-tuned) - 12279 classes       ✅ Yes (12279 classes)
# ====================================================================================================
#
# Returns dictionary: {'model_name': {'description': '...', 'has_trained_classifier': True/False, ...}}
print(f"Available models: {list(models.keys())}")

# Get detailed model information
model_info = describe_model("beats_naturelm", verbose=True)
# Prints formatted output showing:
# - Model type and device
# - Whether it has a trained classifier
# - Number of classes (if applicable)
# - Checkpoint and class mapping paths
# - Audio configuration
# - Usage examples

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

The framework supports a plugin architecture that allows users to register custom model classes without modifying the core library.

**Important**: Registration is only required if you want to use `build_model()` or `build_model_from_spec()` with ModelSpecs. For direct instantiation, registration is not needed.

See [docs/custom_model_registration.md](docs/custom_model_registration.md) for detailed guidance on when and why to register custom models.

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

# Now you can use it with build_model() if you also register a ModelSpec
# Or use it directly without registration: MyCustomModel(device="cpu", num_classes=10)
model = MyCustomModel(num_classes=10, device="cpu")
```

## 🎯 Supported Models

### Official Models

The framework includes support for various audio representation learning models:

- **EfficientNet**: EfficientNet-based models adapted for audio classification
- **BEATs**: BEATs transformer models for audio representation learning
- **EAT**: Efficient Audio Transformer models
- **AVES**: AVES model for bioacoustics
- **BirdMAE**: BirdMAE masked autoencoder for bioacoustic representation learning

### Model Configuration

Models are configured using YAML files which contain the model specifications `model_spec`. The official config files are in the `api/configs/official_models/` directory. These files define the model architecture, audio preprocessing parameters, and optional checkpoint/label mapping paths.

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

These configurations can be loaded directly with `load_model("path/to/config.yml")`. See the "Loading Pre-trained Models" section for usage examples.

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

### Audio Requirements

**Sample Rate**: Each model expects audio at a specific sample rate (defined in its `model_spec`).

**Finding the expected sample rate:**

```python
from representation_learning import describe_model, get_model_spec

# Option 1: Use describe_model() for a formatted overview
describe_model("beats_naturelm", verbose=True)
# Prints: 🎵 Sample Rate: 16000 Hz

# Option 2: Access programmatically via get_model_spec()
spec = get_model_spec("beats_naturelm")
target_sr = spec.audio_config.sample_rate  # 16000
```

**Resampling audio (using librosa):**

For full reproducibility, use `librosa.resample` with `res_type="kaiser_best", scale=True`.

```python
import librosa
import torch
from representation_learning import get_model_spec, load_model

# Get the model's expected sample rate
spec = get_model_spec("beats_naturelm")
target_sr = spec.audio_config.sample_rate

# Load audio at original sample rate
audio, original_sr = librosa.load("audio.wav", sr=None)

# Resample if needed (use these exact parameters for reproducibility)
if original_sr != target_sr:
    audio = librosa.resample(
        audio,
        orig_sr=original_sr,
        target_sr=target_sr,
        res_type="kaiser_best",
        scale=True,
    )

# Convert to tensor and add batch dimension
audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()  # Shape: (1, num_samples)

# Run inference
model = load_model("beats_naturelm", return_features_only=True, device="cpu")
with torch.no_grad():
    output = model(audio_tensor, padding_mask=None)
```

> **Note**: The models were trained with audio resampled using `res_type="kaiser_best"` and `scale=True`. Using different resampling methods may affect results.

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

## 🚀 Training and Evaluation with the API

### Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from representation_learning import build_model

# Create a backbone for training (attach your own head or use a probe)
model = build_model("efficientnet", device="cpu")

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
model = load_model("sl_beats_animalspeak", device="cpu")
model.eval()

# Run inference
with torch.no_grad():
    audio = torch.randn(1, 16000 * 5)  # 5 seconds of audio
    outputs = model(audio, padding_mask=None)
    predictions = torch.softmax(outputs, dim=-1)

    # If model has label mapping, get human-readable labels
    if hasattr(model, "label_mapping"):
        top_k = 5
        probs, indices = torch.topk(predictions, top_k)
        for prob, idx in zip(probs[0], indices[0]):
            label = model.label_mapping["index_to_label"][idx.item()]
            print(f"{label}: {prob.item():.4f}")
```

For complete evaluation examples, see `examples/05_training_and_evaluation.py`.

## 🎨 Embedding Extraction and Feature Representations

### Understanding `return_features_only=True`

When loading models with `return_features_only=True`, the model returns **unpooled features** instead of classification logits. This preserves temporal and spatial information, providing richer representations for downstream tasks.

```python
# Load model for embedding extraction
model = load_model("beats_naturelm", return_features_only=True, device="cpu")
model.eval()

# Get unpooled features
audio = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
features = model(audio, padding_mask=None)
# features.shape = (batch, time_steps, feature_dim)
```

### Model-Specific Output Formats

Different models return features in different formats when `return_features_only=True`:

#### BEATs (Bidirectional Encoder representation from Audio Transformers)

**Output Shape**: `(batch, time_steps, 768)`

**Key Characteristics**:
- Each time step contains **8 embeddings** (one per frequency band)
- Structure: `[T0_0, T0_1, T0_2, T0_3, T0_4, T0_5, T0_6, T0_7, T1_0, T1_1, ...]`
- **Frame rate**: 6.25 Hz (not 100 Hz)
  - Calculated as: 100 Hz / 16 (patch embedding size) = 6.25 Hz
  - For 16 kHz input audio
- Feature dimension: 768 per embedding

**Example**:
```python
model = load_model("beats_naturelm", return_features_only=True, device="cpu")
audio = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
features = model(audio, padding_mask=None)
# features.shape = (1, ~31, 768)
# 31 frames ≈ 5 seconds * 6.25 Hz
# Each frame has 768-dimensional features representing 8 frequency bands
```

**Understanding BEATs Frame Structure**:
- Audio at 16 kHz: 16,000 samples per second
- Patch embedding size: 16 samples
- Base frame rate: 100 Hz (1000 ms / 10 ms per frame)
- Actual frame rate after patching: 100 Hz / 16 = **6.25 Hz**
- Each frame covers: 1 / 6.25 = **160 ms** of audio
- For 5 seconds of audio: 5 * 6.25 = **31.25 frames**

**Use Cases**:
```python
# Option 1: Pool manually for classification
pooled = features.mean(dim=1)  # (batch, 768)

# Option 2: Use specific frequency band
band_0 = features[:, :, :96]  # First frequency band (assuming 96-dim per band)

# Option 3: Use for sequence modeling
# Features preserve temporal structure for RNNs, Transformers, etc.
```

#### EAT (Efficient Audio Transformer)

**Output Shape**: `(batch, num_patches, 768)`

**Key Characteristics**:
- Returns unpooled patch embeddings from transformer backbone
- Includes CLS token as first patch (index 0)
- Number of patches depends on input length and patch size
- Feature dimension: 768 per patch

**Example**:
```python
model = load_model("sl_eat_animalspeak_ssl_all", return_features_only=True, device="cpu")
audio = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
features = model(audio, padding_mask=None)
# features.shape = (1, 513, 768)
# 513 patches = 1 CLS token + 512 spectrogram patches
```

**Use Cases**:
```python
# Option 1: Use CLS token (typically most informative)
cls_token = features[:, 0]  # (batch, 768)

# Option 2: Mean pooling over all patches
pooled = features.mean(dim=1)  # (batch, 768)

# Option 3: Exclude CLS token and pool
spatial_features = features[:, 1:]  # Exclude CLS token
pooled = spatial_features.mean(dim=1)  # (batch, 768)
```

#### EfficientNet

**Output Shape**: `(batch, channels, height, width)`

**Key Characteristics**:
- Returns spatial feature maps before global average pooling
- Preserves 2D spatial structure of spectrogram
- Channel and spatial dimensions depend on model variant

**Example**:
```python
model = load_model("efficientnet", num_classes=10, return_features_only=True, device="cpu")
audio = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
features = model(audio, padding_mask=None)
# features.shape = (1, 1280, 4, 5) for EfficientNet-B0
# 1280 channels, 4x5 spatial dimensions
```

**Use Cases**:
```python
# Option 1: Global average pooling
pooled = features.mean(dim=[2, 3])  # (batch, 1280)

# Option 2: Max pooling
pooled = features.amax(dim=[2, 3])  # (batch, 1280)

# Option 3: Flatten for spatial awareness
flattened = features.flatten(1)  # (batch, 1280*4*5)
```


See `examples/06_embedding_extraction.py` for comprehensive examples of embedding extraction with different models.

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
| `06_embedding_extraction.py` | Feature extraction with `return_features_only=True` (unpooled features) |
| `07_classifier_head_loading.py` | Understanding classifier head behavior with different `num_classes` settings |
| `colab_sl_beats_demo.ipynb` | Google Colab demo for the sl-beats model |

### Custom Model Registration

```python
import torch
import torch.nn as nn
from representation_learning.models.base_model import ModelBase
from representation_learning import register_model_class

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
model = MyAudioCNN(device="cpu", num_classes=10)
```

### Loading Pre-trained Models

**Checkpoint Path Management**

Checkpoint paths are now managed directly in YAML configuration files (`api/configs/official_models/*.yml`). The framework reads checkpoint paths from YAML when needed, eliminating the need for a separate checkpoint registry.

**Creating Custom Model Configurations**

To create your own model configuration, create a YAML file with the following structure:

```yaml
# my_model.yml - Custom model configuration
# Optional: Default checkpoint path (can be local or cloud storage)
checkpoint_path: gs://my-bucket/models/my_model.pt

# Optional: Path to label mapping JSON file
class_mapping_path: gs://my-bucket/models/my_model_labels.json

# Required: Model specification
model_spec:
  name: efficientnet  # Model architecture type
  pretrained: false
  device: cuda
  audio_config:
    sample_rate: 16000
    representation: mel_spectrogram
    n_mels: 128
    target_length_seconds: 10
    window_selection: random
  # Model-specific parameters
  efficientnet_variant: b0
```

**Using Custom Configurations**

```python
from representation_learning import load_model, get_checkpoint_path

# Load model from custom YAML file
model = load_model("path/to/my_model.yml")

# Or for official models, checkpoint paths are read automatically from YAML
checkpoint = get_checkpoint_path("efficientnet_animalspeak")
print(f"Default checkpoint: {checkpoint}")

# Load with default checkpoint (from YAML)
# num_classes=None automatically extracts num_classes from checkpoint
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