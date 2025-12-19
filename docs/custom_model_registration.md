# Custom Model Registration

This guide explains when and why you need to register custom model classes with the framework.

## Quick Answer: Do I Need to Register?

**You only need to register your model class if:**
- ✅ You want to use `build_model()` or `build_model_from_spec()` with a `ModelSpec` that references your custom model
- ✅ You want to use the plugin architecture where models are built from configuration files (YAML)

**You do NOT need to register if:**
- ❌ You're just instantiating your model directly: `MyModel(device="cpu", num_classes=10)`
- ❌ You're using your model standalone or attaching probes directly

## When Registration is Required

Registration is required when you want to use the **plugin architecture** - building models from `ModelSpec` configurations:

```python
from representation_learning import build_model, register_model_class
from representation_learning.configs import ModelSpec, AudioConfig
from representation_learning.models.base_model import ModelBase

# 1. Register your custom model class
@register_model_class
class MyCustomModel(ModelBase):
    name = "my_custom_model"  # This name is used for lookup

    def __init__(self, device: str, num_classes: int, **kwargs):
        super().__init__(device=device)
        # Your model implementation
        pass

# 2. Register a ModelSpec that references your model class
from representation_learning import register_model
model_spec = ModelSpec(
    name="my_custom_model",  # Must match the class name above
    pretrained=False,
    device="cpu",
    audio_config=AudioConfig(sample_rate=16000)
)
register_model("my_model_config", model_spec)

# 3. Now you can use build_model() with the ModelSpec
model = build_model("my_model_config", device="cpu", num_classes=10)
```

**Why this is useful:**
- Allows building models from YAML configuration files
- Enables dynamic model selection based on configuration
- Supports the full plugin architecture workflow
- Makes models discoverable via `list_model_classes()`

## When Registration is NOT Required

If you're just using your custom model directly, you don't need to register it:

```python
from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.utils import build_probe_from_config_online
from representation_learning.configs import ProbeConfig

# Define your model (no registration needed)
class MyCustomModel(ModelBase):
    def __init__(self, device: str, num_classes: int):
        super().__init__(device=device)
        # Your model implementation
        pass

# Use it directly
model = MyCustomModel(device="cpu", num_classes=10)

# Attach a probe if needed
probe_config = ProbeConfig(
    probe_type="linear",
    target_layers=["last_layer"],
    aggregation="mean",
    freeze_backbone=True,
)
model_with_probe = build_probe_from_config_online(
    probe_config=probe_config,
    base_model=model,
    num_classes=10,
    device="cpu",
)
```

**This approach is simpler and sufficient for:**
- Standalone model usage
- Direct instantiation
- Attaching probes to custom models
- One-off experiments

## Summary: Registration Decision Tree

```
Do you want to use build_model() or build_model_from_spec()?
│
├─ YES → Do you have a ModelSpec that references your model?
│   │
│   ├─ YES → Register your model class (required)
│   │   └─ Use: @register_model_class decorator
│   │
│   └─ NO → Create a ModelSpec first, then register both
│
└─ NO → Registration not needed
    └─ Just instantiate your model directly
```

## Key Points

1. **Registration enables the plugin architecture**: It allows `build_model()` and `build_model_from_spec()` to find and instantiate your model class by name.

2. **Direct instantiation is always possible**: You can always create your model directly without registration - registration is only needed for the plugin system.

3. **ModelSpec vs Model Class**:
   - `ModelSpec` = configuration (architecture params, audio config, etc.)
   - Model Class = the actual PyTorch model implementation
   - Registration links them together so `build_model()` can find your class when given a ModelSpec

4. **Probes work with both**: Whether you register or not, you can always attach probes to your custom models using `build_probe_from_config_online()` or `build_probe_from_config_offline()`.

## Example: When to Register

**Scenario 1: Building from YAML config (registration required)**
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

# Now you can load from YAML
from representation_learning import load_model
model = load_model("config.yaml", device="cpu")
```

**Scenario 2: Direct usage (registration not needed)**
```python
# Just use your model directly
class MyCustomModel(ModelBase):
    # ...

model = MyCustomModel(device="cpu", num_classes=10)
# No registration needed!
```

