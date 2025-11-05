"""
Model registry for managing available model configurations.

This module provides a centralized registry for model configurations,
automatically loading official models and allowing custom model registration.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Type, Union

from representation_learning.configs import ModelSpec

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a standard dep in this repo
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)

# Private global model registry
_MODEL_REGISTRY: Dict[str, ModelSpec] = {}

# Private global checkpoint registry for default checkpoint paths
_CHECKPOINT_REGISTRY: Dict[str, str] = {}

# Private global model class registry for dynamic model registration
_MODEL_CLASSES: Dict[str, Type] = {}

# Path to official model configurations directory
# Calculated relative to this file: go up from utils/ to models/ to
# representation_learning/, then into configs/official_models/
_OFFICIAL_MODELS_DIR = (
    Path(__file__).resolve().parents[2] / "configs" / "official_models"
)


def _auto_register_from_yaml(config_dir: Path) -> None:
    """Automatically load YAML configs as ModelSpec objects."""
    if not config_dir.exists():
        logger.warning(f"Config directory does not exist: {config_dir}")
        return

    registered_count = 0
    for yml_path in config_dir.glob("*.yml"):
        name = yml_path.stem
        try:
            spec = load_model_spec_from_yaml(yml_path)
            _MODEL_REGISTRY[name] = spec
            registered_count += 1
            logger.debug(f"Registered model: {name}")
        except Exception as e:
            logger.exception(
                f"Failed to register model config from {yml_path.name}: {e}"
            )

    logger.info(f"Auto-registered {registered_count} models from {config_dir}")


def initialize_registry() -> None:
    """Initialize built-in registry from packaged configs."""
    logger.info(f"Initializing model registry from: {_OFFICIAL_MODELS_DIR}")
    _auto_register_from_yaml(_OFFICIAL_MODELS_DIR)

    logger.info(
        f"Model registry initialized with {len(_MODEL_REGISTRY)} models: "
        f"{list(_MODEL_REGISTRY.keys())}"
    )


def load_model_spec_from_yaml(yaml_path: Union[str, Path]) -> ModelSpec:
    """Load a ModelSpec directly from YAML.

    This function extracts the model specification from YAML files, supporting:
    1) Files with top-level `model_spec: {...}` key (e.g., full RunConfig files)
    2) Files with ModelSpec fields directly at the root level

    Args:
        yaml_path: Path to a YAML file containing a model definition

    Returns:
        A validated ModelSpec instance

    Raises:
        ValueError: If YAML cannot be parsed into a ModelSpec
    """
    if yaml is None:
        raise ValueError("PyYAML not available to parse YAML files")

    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("YAML must define a mapping for model specification")

    # Accept either top-level model_spec or direct ModelSpec fields at root
    model_dict = data.get("model_spec", data)
    if not isinstance(model_dict, dict):
        raise ValueError("Invalid model specification structure in YAML")

    try:
        return ModelSpec(**model_dict)
    except Exception as e:
        raise ValueError(f"Failed to build ModelSpec from YAML: {e}") from e


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

    if name in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' is already registered. Use update_model() to overwrite."
        )

    _MODEL_REGISTRY[name] = model_spec
    logger.info(f"Registered custom model: {name}")


def update_model(name: str, model_spec: ModelSpec) -> None:
    """Update an existing model spec.

    Args:
        name: Name of the model to update
        model_spec: New ModelSpec configuration object
    """
    # Initialize first (without lock to avoid deadlock)
    ensure_initialized()

    _MODEL_REGISTRY[name] = model_spec
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

    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered")

    del _MODEL_REGISTRY[name]
    logger.info(f"Unregistered model: {name}")


def get_model(name: str) -> Optional[ModelSpec]:
    """Get a model spec by name.

    Args:
        name: Name of the model

    Returns:
        ModelSpec if found, None otherwise
    """
    ensure_initialized()
    return _MODEL_REGISTRY.get(name)


def list_models() -> Dict[str, ModelSpec]:
    """Return available registered models.

    Returns:
        Copy of the model registry
    """
    ensure_initialized()
    return _MODEL_REGISTRY.copy()


def list_model_names() -> list[str]:
    """Return list of registered model names.

    Returns:
        List of model names
    """
    ensure_initialized()
    return list(_MODEL_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if a model is registered.

    Args:
        name: Name of the model to check

    Returns:
        True if registered, False otherwise
    """
    ensure_initialized()
    return name in _MODEL_REGISTRY


def register_checkpoint(name: str, checkpoint_path: str) -> None:
    """Register a default checkpoint path for a model.

    Args:
        name: Name of the model
        checkpoint_path: Default checkpoint path
    """
    _CHECKPOINT_REGISTRY[name] = checkpoint_path
    logger.info(f"Registered default checkpoint for '{name}': {checkpoint_path}")


def get_checkpoint(name: str) -> Optional[str]:
    """Get default checkpoint path for a model.

    Args:
        name: Name of the model

    Returns:
        Checkpoint path if found, None otherwise
    """
    return _CHECKPOINT_REGISTRY.get(name)


def unregister_checkpoint(name: str) -> None:
    """Remove default checkpoint path for a model.

    Args:
        name: Name of the model

    Raises:
        KeyError: If checkpoint is not registered
    """
    if name not in _CHECKPOINT_REGISTRY:
        raise KeyError(f"No default checkpoint registered for model '{name}'")

    del _CHECKPOINT_REGISTRY[name]
    logger.info(f"Unregistered default checkpoint for: {name}")


def describe_model(name: str, verbose: bool = False) -> dict:
    """Return a detailed summary of the model configuration.

    Args:
        name: Name of the model to describe
        verbose: If True, pretty-print the model information to stdout

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

    if verbose:
        try:
            import json

            print(json.dumps(model_info, indent=2, sort_keys=True, default=str))
        except Exception:
            # Fallback to repr if something is not JSON-serializable
            print(model_info)

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

    if name in _MODEL_CLASSES:
        logger.warning(f"Model class '{name}' is already registered, overwriting.")
    _MODEL_CLASSES[name] = cls
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
    if name not in _MODEL_CLASSES:
        raise KeyError(f"Model class '{name}' is not registered.")
    return _MODEL_CLASSES[name]


def list_model_classes() -> list[str]:
    """Return a list of registered model class names.

    Returns:
        List of registered model class names
    """
    return list(_MODEL_CLASSES.keys())


def is_model_class_registered(name: str) -> bool:
    """Check if a model class is registered.

    Args:
        name: Name of the model class to check

    Returns:
        True if registered, False otherwise
    """
    return name in _MODEL_CLASSES


def unregister_model_class(name: str) -> None:
    """Remove a model class from the registry.

    Args:
        name: Name of the model class to remove

    Raises:
        KeyError: If model class is not registered
    """
    if name not in _MODEL_CLASSES:
        raise KeyError(f"Model class '{name}' is not registered")

    del _MODEL_CLASSES[name]
    logger.info(f"Unregistered model class: {name}")


def ensure_initialized() -> None:
    """Ensure the registry is initialized before use."""
    if not _MODEL_REGISTRY:
        initialize_registry()
