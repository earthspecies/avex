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
