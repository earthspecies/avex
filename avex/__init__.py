"""
Representation Learning Framework

A comprehensive Python-based system for training, evaluating, and analyzing
audio representation learning models with support for both supervised and
self-supervised learning paradigms.
"""

from importlib.metadata import version

from .models.utils.factory import (
    build_model,
    build_model_from_spec,
)
from .models.utils.load import load_label_mapping, load_model
from .models.utils.registry import (
    describe_model,
    get_checkpoint_path,
    get_model_class,
    get_model_spec,
    list_model_classes,
    list_model_layers,
    list_models,
    register_model,
    register_model_class,
)

__version__ = version("avex")


__all__ = [
    # Model loading
    "load_model",
    # Registry management
    "register_model",
    "get_model_spec",
    "list_models",
    "describe_model",
    "list_model_layers",
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
