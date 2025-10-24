"""
Model registry for managing available model configurations.

This module provides a centralized registry for model configurations,
automatically loading official models and allowing custom model registration.
"""

import logging
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Type

from representation_learning.configs import ModelSpec, RunConfig

logger = logging.getLogger(__name__)

# Global model registry
MODEL_REGISTRY: Dict[str, ModelSpec] = {}

# Global checkpoint registry for default checkpoint paths
CHECKPOINT_REGISTRY: Dict[str, str] = {}

# Global model class registry for dynamic model registration
MODEL_CLASSES: Dict[str, Type] = {}

# Thread safety lock for registry operations
_registry_lock = Lock()


def _auto_register_from_yaml(config_dir: Path) -> None:
    """Automatically load YAML configs as ModelSpec objects."""
    if not config_dir.exists():
        logger.warning(f"Config directory does not exist: {config_dir}")
        return

    registered_count = 0
    for yml_path in config_dir.glob("*.yml"):
        name = yml_path.stem
        try:
            run_cfg = RunConfig.from_sources(yaml_file=yml_path, cli_args=())
            MODEL_REGISTRY[name] = run_cfg.model_spec
            registered_count += 1
            logger.debug(f"Registered model: {name}")
        except Exception as e:
            logger.exception(
                f"Failed to register model config from {yml_path.name}: {e}"
            )

    logger.info(f"Auto-registered {registered_count} models from {config_dir}")


def initialize_registry() -> None:
    """Initialize built-in registry from packaged configs."""
    # Get the package root directory (go up from utils/ to models/ to
    # representation_learning/)
    root = Path(__file__).resolve().parents[2]
    official_dir = root / "configs" / "official_models"

    logger.info(f"Initializing model registry from: {official_dir}")
    _auto_register_from_yaml(official_dir)

    logger.info(
        f"Model registry initialized with {len(MODEL_REGISTRY)} models: "
        f"{list(MODEL_REGISTRY.keys())}"
    )


def register_model(name: str, model_spec: ModelSpec) -> None:
    """Register a new model spec dynamically.

    Args:
        name: Unique name for the model
        model_spec: ModelSpec configuration object

    Raises:
        ValueError: If name is already registered
    """
    # Initialize first (without lock to avoid deadlock)
    ensure_initialized()

    with _registry_lock:
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered. Use update_model() to "
                f"overwrite."
            )

        MODEL_REGISTRY[name] = model_spec
        logger.info(f"Registered custom model: {name}")


def update_model(name: str, model_spec: ModelSpec) -> None:
    """Update an existing model spec.

    Args:
        name: Name of the model to update
        model_spec: New ModelSpec configuration object
    """
    # Initialize first (without lock to avoid deadlock)
    ensure_initialized()

    with _registry_lock:
        MODEL_REGISTRY[name] = model_spec
        logger.info(f"Updated model: {name}")


def unregister_model(name: str) -> None:
    """Remove a model from the registry.

    Args:
        name: Name of the model to remove

    Raises:
        KeyError: If model is not registered
    """
    # Initialize first (without lock to avoid deadlock)
    ensure_initialized()

    with _registry_lock:
        if name not in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is not registered")

        del MODEL_REGISTRY[name]
        logger.info(f"Unregistered model: {name}")


def get_model(name: str) -> Optional[ModelSpec]:
    """Get a model spec by name.

    Args:
        name: Name of the model

    Returns:
        ModelSpec if found, None otherwise
    """
    ensure_initialized()
    return MODEL_REGISTRY.get(name)


def list_models() -> Dict[str, ModelSpec]:
    """Return available registered models.

    Returns:
        Copy of the model registry
    """
    ensure_initialized()
    return MODEL_REGISTRY.copy()


def list_model_names() -> list[str]:
    """Return list of registered model names.

    Returns:
        List of model names
    """
    ensure_initialized()
    return list(MODEL_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if a model is registered.

    Args:
        name: Name of the model to check

    Returns:
        True if registered, False otherwise
    """
    ensure_initialized()
    return name in MODEL_REGISTRY


def register_checkpoint(name: str, checkpoint_path: str) -> None:
    """Register a default checkpoint path for a model.

    Args:
        name: Name of the model
        checkpoint_path: Default checkpoint path
    """
    with _registry_lock:
        CHECKPOINT_REGISTRY[name] = checkpoint_path
        logger.info(f"Registered default checkpoint for '{name}': {checkpoint_path}")


def get_checkpoint(name: str) -> Optional[str]:
    """Get default checkpoint path for a model.

    Args:
        name: Name of the model

    Returns:
        Checkpoint path if found, None otherwise
    """
    return CHECKPOINT_REGISTRY.get(name)


def unregister_checkpoint(name: str) -> None:
    """Remove default checkpoint path for a model.

    Args:
        name: Name of the model

    Raises:
        KeyError: If checkpoint is not registered
    """
    with _registry_lock:
        if name not in CHECKPOINT_REGISTRY:
            raise KeyError(f"No default checkpoint registered for model '{name}'")

        del CHECKPOINT_REGISTRY[name]
        logger.info(f"Unregistered default checkpoint for: {name}")


def describe_model(name: str) -> dict:
    """Return a detailed summary of the model configuration.

    Args:
        name: Name of the model to describe

    Returns:
        Dictionary containing the model's configuration details

    Raises:
        KeyError: If model is not found in registry
    """
    ensure_initialized()
    spec = get_model(name)
    if not spec:
        raise KeyError(f"Model '{name}' not found in registry")

    # Get the full model dump with all fields
    model_info = spec.model_dump()

    # Add some additional metadata
    model_info["_metadata"] = {
        "name": name,
        "model_type": spec.name,
        "pretrained": spec.pretrained,
        "device": spec.device,
        "has_audio_config": spec.audio_config is not None,
        "has_text_model": spec.text_model_name is not None,
        "has_eat_config": spec.eat_cfg is not None,
        "is_pretraining_mode": spec.pretraining_mode,
    }

    return model_info


def register_model_class(cls: Type) -> Type:
    """
    Register a new model class for use in the framework.

    The `name` property on the class is used as the key. If not set, the class
    name lowercased will be used.

    Args:
        cls: Model class that inherits from ModelBase

    Returns:
        The registered class (for use as decorator)

    """
    name = getattr(cls, "name", cls.__name__.lower())

    with _registry_lock:
        if name in MODEL_CLASSES:
            logger.warning(f"Model class '{name}' is already registered, overwriting.")
        MODEL_CLASSES[name] = cls
        logger.info(f"Registered model class: {name}")

    return cls


def get_model_class(name: str) -> Type:
    """
    Get a registered model class by name.

    Args:
        name: Name of the model class

    Returns:
        The registered model class

    Raises:
        KeyError: If the model class is not registered
    """
    with _registry_lock:
        if name not in MODEL_CLASSES:
            raise KeyError(f"Model class '{name}' is not registered.")
        return MODEL_CLASSES[name]


def list_model_classes() -> list[str]:
    """Return a list of registered model class names.

    Returns:
        List of registered model class names
    """
    with _registry_lock:
        return list(MODEL_CLASSES.keys())


def is_model_class_registered(name: str) -> bool:
    """Check if a model class is registered.

    Args:
        name: Name of the model class to check

    Returns:
        True if registered, False otherwise
    """
    with _registry_lock:
        return name in MODEL_CLASSES


def unregister_model_class(name: str) -> None:
    """Remove a model class from the registry.

    Args:
        name: Name of the model class to remove

    Raises:
        KeyError: If model class is not registered
    """
    with _registry_lock:
        if name not in MODEL_CLASSES:
            raise KeyError(f"Model class '{name}' is not registered")

        del MODEL_CLASSES[name]
        logger.info(f"Unregistered model class: {name}")


def ensure_initialized() -> None:
    """Ensure the registry is initialized before use."""
    with _registry_lock:
        if not MODEL_REGISTRY:
            initialize_registry()
