# Custom Model Registration Guide

This guide walks you through using custom models with the representation learning framework, starting with the simplest approach and building up to more advanced use cases.

**Most of the time, you don't need to register your custom model.** You can use it directly:

```python
from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.utils import build_probe_from_config
from representation_learning.configs import ProbeConfig

# Define your model
class MyCustomModel(ModelBase):
    def __init__(self, device: str, num_classes: int):
        super().__init__(device=device)
        # Your model implementation
        pass

# Use it directly - no registration needed!
model = MyCustomModel(device="cpu", num_classes=10)

```

**This simple approach works for:**
- Standalone model usage
- Direct instantiation
- One-off experiments

## Quick Reference

**Do I need to register?**

```
How do you want to use your custom model?
│
├─ (1) Direct instantiation: MyModel(device="cpu", num_classes=10)
│   └─ No registration needed
│
└─ (2) Plugin architecture: build_model() or build_model_from_spec()
    └─ Registration required: Use @register_model_class decorator
```

## Tutorial: Using the Plugin Architecture

If you want to use the plugin system, follow these steps:

### Step 1: Register Your Model Class

```python
from representation_learning import register_model_class
from representation_learning.models.base_model import ModelBase

@register_model_class
class MyCustomModel(ModelBase):
    name = "my_custom_model"  # This name is used for lookup

    def __init__(self, device: str, num_classes: int, **kwargs):
        super().__init__(device=device)
        # Your model implementation
        pass
```

### Step 2: Create and Use a ModelSpec

```python
from representation_learning.configs import ModelSpec, AudioConfig
from representation_learning.models.utils.factory import build_model_from_spec

# Create a ModelSpec that references your model class
model_spec = ModelSpec(
    name="my_custom_model",  # Must match the class name above
    pretrained=False,
    device="cpu",
    audio_config=AudioConfig(sample_rate=16000)
)

# Use the ModelSpec to build your model
model = build_model_from_spec(model_spec, device="cpu", num_classes=10)
```

**Note:** Creating a `ModelSpec` doesn't validate that the model class exists. The check happens when you call `build_model_from_spec()`, which will raise a `KeyError` if the model class isn't registered.

### Step 3 (Optional): Register the ModelSpec for Reuse

If you want to reuse the same configuration, you can register it:

```python
from representation_learning import register_model, build_model

register_model("my_model_config", model_spec)

# Now you can use the registered name
model = build_model("my_model_config", device="cpu", num_classes=10)
```

## Key Concepts

- **Model Class**: Your PyTorch model implementation (inherits from `ModelBase`)
- **ModelSpec**: Configuration object (architecture params, audio config, etc.)
- **Registration**: Links your model class to the plugin system so it can be found by name

## Advanced: Loading from YAML

If you want to load models from YAML configuration files:

```python
# config.yaml
name: my_custom_model
pretrained: false
audio_config:
  sample_rate: 16000

# Python code
@register_model_class
class MyCustomModel(ModelBase):
    name = "my_custom_model"
    # ...

# Load from YAML
from representation_learning import load_model
model = load_model("config.yaml", device="cpu")
```

## Loading Pre-trained Models

### Checkpoint Path Management

Checkpoint paths are now managed directly in YAML configuration files (`representation_learning/api/configs/official_models/*.yml`). The framework reads checkpoint paths from YAML when needed, eliminating the need for a separate checkpoint registry.

### Creating Custom Model Configurations

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

### Using Custom Configurations

```python
from representation_learning import load_model, get_checkpoint_path

# Load model from custom YAML file
model = load_model("path/to/my_model.yml")

# Or for official models, checkpoint paths are read automatically from YAML
checkpoint = get_checkpoint_path("efficientnet_animalspeak")
print(f"Default checkpoint: {checkpoint}")

# Load with default checkpoint (from YAML)
model = load_model("efficientnet_animalspeak")  # Uses YAML checkpoint

# Load with custom checkpoint (overrides YAML default)
# Priority: user-provided checkpoint_path > YAML default > no checkpoint
model = load_model("efficientnet_animalspeak", checkpoint_path="gs://my-custom-checkpoint.pt")

# Load for embedding extraction (strip classifier head when present)
base = load_model("efficientnet_animalspeak", return_features_only=True)
```

### Checkpoint Path Priority

When loading a model, checkpoint paths are resolved in this order:
1. **User-provided `checkpoint_path` parameter** (highest priority)
2. **Default checkpoint from YAML file**
3. **No checkpoint** (for embedding extraction or new models)

### Classifier Head Behavior

- `load_model()` preserves a trained classifier head when it is present in the checkpoint.
- To build a new classifier for a new task, load a backbone with `return_features_only=True`
  and attach a probe head via `build_probe_from_config()` (see probe documentation).

### `pretrained=True` Without Checkpoint

When `pretrained=True` and no `checkpoint_path` is set:
- The model uses its own pretrained weight loading mechanism (varies by model type)
- BEATs: Loads from hardcoded SSL/ImageNet paths
- EfficientNet: Loads ImageNet weights via torchvision
- EAT-HF: Loads from HuggingFace
- **Note**: If a `checkpoint_path` is found (from YAML or user-provided), `pretrained` is automatically set to `False` to prioritize checkpoint weights
