"""
Core API for the representation learning framework.

This module provides the main user-facing API for loading models and managing
the model registry. It re-exports functionality from the models package.
"""

from representation_learning.models.utils.factory import (
    build_model,
    build_model_from_spec,
)
from representation_learning.models.utils.load import create_model, load_model
from representation_learning.models.utils.registry import (
    describe_model,
    get_checkpoint,
    get_model,
    get_model_class,
    is_model_class_registered,
    is_registered,
    list_model_classes,
    list_model_names,
    list_models,
    register_checkpoint,
    register_model,
    register_model_class,
    unregister_checkpoint,
    unregister_model,
    unregister_model_class,
    update_model,
)

__all__ = [
    # Model loading
    "load_model",
    "create_model",
    # Registry management
    "register_model",
    "update_model",
    "unregister_model",
    "get_model",
    "list_models",
    "list_model_names",
    "is_registered",
    "describe_model",
    # Model class management
    "register_model_class",
    "get_model_class",
    "list_model_classes",
    "is_model_class_registered",
    "unregister_model_class",
    # Model factory
    "build_model",
    "build_model_from_spec",
    # Checkpoint management
    "register_checkpoint",
    "get_checkpoint",
    "unregister_checkpoint",
]
