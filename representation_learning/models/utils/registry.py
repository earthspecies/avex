"""
Model registry for managing available model configurations.

This module provides a centralized registry for model configurations,
automatically loading official models and allowing custom model registration.
"""

import logging
from importlib import resources
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

# Private global model class registry for dynamic model registration
_MODEL_CLASSES: Dict[str, Type] = {}

# Package containing official model YAML configurations
_OFFICIAL_MODELS_PKG = "representation_learning.api.configs.official_models"


def _auto_register_from_yaml() -> None:
    """Automatically load packaged YAML configs as ModelSpec objects.

    Uses importlib.resources.files() to enumerate YAML files from the
    installed package, which works both in editable installs and when the
    package is installed as a dependency (including zipped wheels). YAML is
    read with entry.open() for zip-safety.
    """
    registered_count = 0
    try:
        root = resources.files(_OFFICIAL_MODELS_PKG)
        for entry in root.iterdir():
            if not entry.name.endswith(".yml") or not entry.is_file():
                continue
            name = Path(entry.name).stem
            try:
                # Use as_file to obtain a filesystem path (zip-safe), then reuse
                # load_model_spec_from_yaml for consistent parsing and validation.
                with resources.as_file(entry) as yaml_path:
                    spec = load_model_spec_from_yaml(yaml_path)
                _MODEL_REGISTRY[name] = spec
                registered_count += 1
                logger.debug(f"Registered model: {name}")
            except Exception as e:  # pragma: no cover - defensive
                logger.exception(f"Failed to register model config from {entry.name}: {e}")
        logger.info(f"Auto-registered {registered_count} models from package {_OFFICIAL_MODELS_PKG}")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to load models from package {_OFFICIAL_MODELS_PKG}: {e}")


def initialize_registry() -> None:
    """Initialize built-in registry from packaged configs.

    Checkpoint paths are automatically registered from ModelSpec.checkpoint_path
    in the YAML files.
    """
    logger.info(f"Initializing model registry from package: {_OFFICIAL_MODELS_PKG}")
    _auto_register_from_yaml()

    logger.info(f"Model registry initialized with {len(_MODEL_REGISTRY)} models: {list(_MODEL_REGISTRY.keys())}")


def load_model_spec_from_yaml(yaml_path: Union[str, Path]) -> ModelSpec:
    """Load a ModelSpec directly from YAML.

    This function extracts the model specification from YAML files, supporting:
    1) Files with top-level `model_spec: {...}` key (e.g., full RunConfig files)
    2) Files with ModelSpec fields directly at the root level

    Args:
        yaml_path: Path to a YAML file (str or Path) containing a model
            definition.

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
        raise ValueError(f"Model '{name}' is already registered. Use update_model() to overwrite.")

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


def get_model(name: str) -> ModelSpec:
    """Get a model spec by name.

    Args:
        name: Name of the model

    Returns:
        ModelSpec if found

    Raises:
        KeyError: If model is not registered
    """
    ensure_initialized()
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered")
    return _MODEL_REGISTRY[name]


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


def get_checkpoint_path(name: str) -> Optional[str]:
    """Get default checkpoint path for a model from its YAML configuration.

    Reads checkpoint_path directly from the YAML file associated with the model.
    For official models, this is configs/official_models/{name}.yml

    Args:
        name: Name of the model (must be registered in MODEL_REGISTRY)

    Returns:
        Checkpoint path if found in YAML, None otherwise

    Raises:
        KeyError: If model is not registered
    """
    ensure_initialized()

    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered")

    # For official models, read checkpoint_path from YAML file packaged in resources
    try:
        root = resources.files(_OFFICIAL_MODELS_PKG)
        yaml_file = root / f"{name}.yml"
        if yaml_file.is_file():
            # Use entry.open() for zip-safe reading
            with yaml_file.open("r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
            if isinstance(yaml_data, dict) and "checkpoint_path" in yaml_data:
                checkpoint_path = yaml_data["checkpoint_path"]
                if checkpoint_path:
                    return checkpoint_path
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(
            "Failed to read checkpoint_path from packaged resource for %s: %s",
            name,
            e,
        )

    # Model is registered but no checkpoint_path in YAML
    return None


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
    # get_model() raises KeyError if model is not found, so no need to check for None
    try:
        spec = get_model(name)
    except KeyError:
        raise  # Re-raise to satisfy DOC502 - KeyError is raised by get_model()

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
