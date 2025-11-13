"""
Utilities for model management and loading.

This package contains utilities for model registration, loading, and factory functions.
"""

from .factory import build_model, build_model_from_spec
from .load import create_model, load_model
from .registry import (
    describe_model,
    get_checkpoint_path,
    get_model_class,
    get_model_spec,
    list_model_classes,
    list_models,
    register_model,
    register_model_class,
)

__all__ = [
    # Model factory
    "build_model",
    "build_model_from_spec",
    # Model loading
    "load_model",
    "create_model",
    # Registry management
    "register_model",
    "get_model_spec",
    "list_models",
    "describe_model",
    # Checkpoint management
    "get_checkpoint_path",
    # Model class management
    "register_model_class",
    "get_model_class",
    "list_model_classes",
]
