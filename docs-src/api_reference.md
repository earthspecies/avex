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
from avex import load_model

# Load with custom checkpoint
model = load_model("esp_aves2_effnetb0_all", checkpoint_path="gs://my-bucket/checkpoint.pt")

# Load with default checkpoint (from YAML config)
# Checkpoint paths are defined in `avex/api/configs/official_models/*.yml`
# The loader preserves the classifier head from the checkpoint when present.
model = load_model("efficientnet_animalspeak")  # Uses default checkpoint from YAML

# Load for embedding extraction (no classifier head)
model = load_model("esp_aves2_naturelm_audio_v1_beats", return_features_only=True)  # Returns features, not logits

# Load from config file
model = load_model("experiments/my_model.yml")
```

**Important: `checkpoint_path`, `return_features_only`, and `pretrained` behavior**

- `load_model()` **does not** accept `num_classes`. If you need a new classifier for a
  new task, load a backbone with `return_features_only=True` and attach a probe head
  via `build_probe_from_config()`.

- If `checkpoint_path` is provided (either explicitly or found from YAML),
  the loader disables model-default pretrained weight loading to avoid double-loading.

**`pretrained`**

`pretrained` is a `ModelSpec` field (configured in YAML / `ModelSpec`), not a
`load_model()` argument. When a checkpoint is used, checkpoint weights take priority
over any model-default pretrained weights.

```python
# Load for embedding extraction (no classifier head)
model = load_model("esp_aves2_naturelm_audio_v1_beats", return_features_only=True, device="cpu")

# Load with default checkpoint (from YAML config)
model = load_model("esp_aves2_effnetb0_all", device="cpu")
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
from avex import build_model, register_model_class
from avex.models.base_model import ModelBase

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
from avex import (
    register_model, get_model_spec, list_models, describe_model
)

# Register a new model configuration
from avex.configs import ModelSpec, AudioConfig
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
# esp_aves2_naturelm_audio_v1_beats   BEATs with NatureLM (audio v1)            ✅ Yes
# sl_beats_animalspeak                beats (fine-tuned) - 12279 classes       ✅ Yes (12279 classes)
# ====================================================================================================
#
# Returns dictionary: {'model_name': {'description': '...', 'has_trained_classifier': True/False, ...}}
print(f"Available models: {list(models.keys())}")

# Get detailed model information
model_info = describe_model("esp_aves2_naturelm_audio_v1_beats", verbose=True)
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
from avex import (
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
from avex import get_checkpoint_path

# Get default checkpoint path from YAML config
# Checkpoint paths are defined in `avex/api/configs/official_models/*.yml`
checkpoint = get_checkpoint_path("esp_aves2_effnetb0_all")
print(f"Default checkpoint: {checkpoint}")

# Override default checkpoint by passing checkpoint_path parameter
from avex import load_model
model = load_model("esp_aves2_effnetb0_all", checkpoint_path="gs://my-custom-checkpoint.pt")
```

### Class Mapping Management
```python
from avex import load_label_mapping

# Load class mappings for a model
# Class mappings define the relationship between class labels and indices
class_mapping = load_label_mapping("esp_aves2_sl_beats_all")
if class_mapping:
    label_to_index = class_mapping["label_to_index"]
    index_to_label = class_mapping["index_to_label"]
    print(f"Loaded {len(label_to_index)} classes")
    print(f"Example: {label_to_index['dog']}")  # Get index for 'dog'
    print(f"Example: {index_to_label[0]}")  # Get label for index 0
```
