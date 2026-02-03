# Architecture

## Core Components

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

## Plugin Architecture

The framework supports a plugin architecture that allows users to register custom model classes without modifying the core library.

**Important**: Registration is only required if you want to use `build_model()` or `build_model_from_spec()` with ModelSpecs. For direct instantiation, registration is not needed.

See {doc}`Custom Model Registration <custom_model_registration>` for detailed guidance on when and why to register custom models.

```python
from avex.models.base_model import ModelBase
from avex import register_model_class

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
