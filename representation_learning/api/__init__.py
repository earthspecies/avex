"""
Representation Learning Framework

A comprehensive Python-based system for training, evaluating, and analyzing
audio representation learning models with support for both supervised and
self-supervised learning paradigms.
"""

from .core import (
    build_model,
    build_model_from_spec,
    create_model,
    describe_model,
    get_checkpoint_path,
    get_model,
    get_model_class,
    is_model_class_registered,
    is_registered,
    list_model_classes,
    list_model_names,
    list_models,
    load_model,
    register_model,
    register_model_class,
    unregister_model,
    unregister_model_class,
    update_model,
)

__version__ = "0.1.0"

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
    "get_checkpoint_path",
]
