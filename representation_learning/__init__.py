"""
Representation Learning Framework

A comprehensive Python-based system for training, evaluating, and analyzing
audio representation learning models with support for both supervised and
self-supervised learning paradigms.
"""

# Initialize CUDA FIRST before any other imports to prevent CUDA from becoming
# unavailable during model imports in environments where CUDA initialization timing matters
# The CUDA error is a byproduct of the fact that the registry is imported in the __init__.py file
try:
    import torch

    # Force CUDA initialization by creating a tensor on CUDA device
    # This "locks in" the CUDA context before any model imports
    if torch.cuda.is_available():
        _ = torch.cuda.device_count()  # Check device count
        # Create a small tensor on CUDA to fully initialize the context
        # This prevents CUDA from becoming unavailable during subsequent imports
        try:
            _cuda_init_tensor = torch.zeros(1, device="cuda")
            del _cuda_init_tensor  # Clean up immediately
            torch.cuda.synchronize()  # Ensure CUDA operations complete
        except Exception:
            pass  # If CUDA tensor creation fails, continue anyway
except Exception:
    pass  # Ignore if torch is not available

from .models.utils.factory import (
    build_model,
    build_model_from_spec,
)
from .models.utils.load import create_model, load_label_mapping, load_model
from .models.utils.registry import (
    describe_model,
    get_checkpoint_path,
    get_model_class,
    get_model_spec,
    list_model_classes,
    list_models,
    register_model,
    register_model_class,
)

try:
    from importlib.metadata import version

    __version__ = version("representation-learning")
except Exception:
    # Fallback for development or if package not installed
    __version__ = "0.1.0"

__all__ = [
    # Model loading
    "load_model",
    "create_model",
    # Registry management
    "register_model",
    "get_model_spec",
    "list_models",
    "describe_model",
    # Model class management
    "register_model_class",
    "get_model_class",
    "list_model_classes",
    # Model factory
    "build_model",
    "build_model_from_spec",
    # Checkpoint management
    "get_checkpoint_path",
    # Label mapping management
    "load_label_mapping",
]
