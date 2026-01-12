# API Reference

## Model Loading Functions

The framework provides three main functions for working with models, each designed for specific use cases:

### `load_model()` - Load Complete Models

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

### Training New Models

**Recommended pattern:**
- Define a custom model class (subclassing `ModelBase`) with its own classifier head, or
- Build a backbone via `build_model` / `build_model_from_spec` and attach a probe head with `build_probe_from_config` (supports both online and offline modes).

### `build_model()` - Plugin Architecture

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

## Model Registry Functions

### Registry Management
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

### Model Class Management (Plugin Architecture)
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

### Checkpoint Management
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

### Class Mapping Management
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
