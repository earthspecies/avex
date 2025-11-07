"""
Model registry for managing available model configurations.

This module provides a centralized registry for model configurations,
automatically loading official models and allowing custom model registration.
"""

import importlib
import inspect
import logging
from importlib import resources
from pathlib import Path
from typing import Dict, Optional, Type, Union

from representation_learning.configs import ModelSpec
from representation_learning.models.base_model import ModelBase

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


def _discover_model_classes() -> Dict[str, Type[ModelBase]]:
    """Discover all ModelBase subclasses in the models package.

    Dynamically scans the models package directory to find all model classes,
    avoiding the need to manually list each model.

    Returns:
        Dictionary mapping model names to their classes
    """
    discovered = {}

    # Mapping from module path to model name(s)
    # This handles special cases where model name doesn't match module/class name
    MODEL_NAME_MAPPING: Dict[str, list[str]] = {
        "representation_learning.models.beats_model": ["beats"],
        "representation_learning.models.atst_frame.atst_encoder": ["atst"],
        "representation_learning.models.resnet": ["resnet18", "resnet50", "resnet152"],
        "representation_learning.models.aves_model": ["aves_bio"],
    }

    # Directories to exclude from scanning
    EXCLUDED_DIRS = {"utils", "probes", "beats"}  # beats is a subdirectory with implementation details

    # Get the models package path
    try:
        models_pkg = importlib.import_module("representation_learning.models")
        models_path = Path(models_pkg.__file__).parent if models_pkg.__file__ else None
    except ImportError:
        logger.warning("Could not import representation_learning.models package")
        return discovered

    if not models_path or not models_path.exists():
        logger.warning(f"Models package path not found: {models_path}")
        return discovered

    # Scan Python files in the models directory and subdirectories
    for py_file in models_path.rglob("*.py"):
        # Skip excluded directories
        if any(excluded in py_file.parts for excluded in EXCLUDED_DIRS):
            continue

        # Skip __init__.py, get_model.py, and base_model.py
        if py_file.name in ("__init__.py", "get_model.py", "base_model.py"):
            continue

        # Convert file path to module name
        # models_path is representation_learning/models, so parent is representation_learning
        # We need the full module path starting from representation_learning
        try:
            # Get path relative to representation_learning package
            rel_path = py_file.relative_to(models_path.parent)
            # Convert to module name: models/clip.py -> representation_learning.models.clip
            module_name = (
                f"representation_learning.{str(rel_path.with_suffix('')).replace('/', '.').replace('\\\\', '.')}"
            )
        except ValueError:
            # File is not under models_path.parent, skip it
            continue

        try:
            module = importlib.import_module(module_name)

            # Find all ModelBase subclasses in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Skip if not a ModelBase subclass, or if it's ModelBase itself
                if not issubclass(obj, ModelBase) or obj is ModelBase:
                    continue

                # Skip if it's imported from another module (not defined here)
                if obj.__module__ != module_name:
                    continue

                # Get model name(s) for this module
                model_names = MODEL_NAME_MAPPING.get(module_name, [])

                if not model_names:
                    # Infer model name from class or module name
                    if name.lower() == "model":
                        # Class is "Model", infer from module name
                        module_base = module_name.split(".")[-1]
                        if module_base.endswith("_model"):
                            model_name = module_base[:-6]  # Remove "_model"
                        elif module_base == "atst_encoder":
                            model_name = "atst"  # Special case for atst_frame/atst_encoder.py
                        else:
                            model_name = module_base
                        model_names = [model_name]
                    elif name.lower().endswith("model"):
                        # Class name like "CLIPModel" -> "clip"
                        model_name = name.lower()[:-5]  # Remove "model"
                        model_names = [model_name]
                    else:
                        model_names = [name.lower()]

                # Register all model names for this class
                for model_name in model_names:
                    discovered[model_name] = obj
                    logger.debug(f"Discovered model class: {model_name} -> {obj.__name__} from {module_name}")

        except ImportError as e:
            # Some modules may have optional dependencies
            logger.debug(f"Could not import {module_name}: {e}")
        except Exception as e:
            logger.warning(f"Error discovering models in {module_name}: {e}")

    return discovered


def _auto_register_model_classes() -> None:
    """Automatically register built-in model classes from the models/ directory.

    This registers all standard model classes so they can be used via the registry
    without manual registration. Model classes are discovered dynamically by scanning
    the models package for ModelBase subclasses.
    """
    try:
        discovered = _discover_model_classes()

        for model_name, model_class in discovered.items():
            _MODEL_CLASSES[model_name] = model_class

        logger.info(f"Auto-registered {len(discovered)} model classes: {sorted(discovered.keys())}")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to auto-register model classes: {e}")


def initialize_registry() -> None:
    """Initialize built-in registry from packaged configs.

    Checkpoint paths are automatically registered from ModelSpec.checkpoint_path
    in the YAML files. Model classes are also automatically registered.
    """
    if _MODEL_REGISTRY:  # Already initialized
        return

    logger.info(f"Initializing model registry from package: {_OFFICIAL_MODELS_PKG}")
    _auto_register_from_yaml()
    _auto_register_model_classes()

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
        name: Name for the model (will overwrite if already registered)
        model_spec: ModelSpec configuration object
    """
    if name in _MODEL_REGISTRY:
        logger.warning(f"Model '{name}' is already registered. Overwriting with new configuration.")
    else:
        logger.info(f"Registered custom model: {name}")

    _MODEL_REGISTRY[name] = model_spec


def get_model_spec(name: str) -> Optional[ModelSpec]:
    """Get a model spec by name.

    Args:
        name: Name of the model

    Returns:
        ModelSpec if found, None otherwise
    """
    if name not in _MODEL_REGISTRY:
        logger.info(f"Model '{name}' is not registered")
        return None
    return _MODEL_REGISTRY[name]


def list_models() -> Dict[str, ModelSpec]:
    """Return available registered models.

    Returns:
        Copy of the model registry
    """
    return _MODEL_REGISTRY.copy()


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
    spec = get_model_spec(name)
    if spec is None:
        raise KeyError(f"Model '{name}' is not registered")

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


def get_model_class(name: str) -> Optional[Type[ModelBase]]:
    """Get a registered model class by name.

    Args:
        name: Name of the model class

    Returns:
        The registered model class if found, None otherwise
    """
    if name not in _MODEL_CLASSES:
        logger.info(f"Model class '{name}' is not registered")
        return None
    return _MODEL_CLASSES[name]


def list_model_classes() -> list[str]:
    """Return a list of registered model class names.

    Returns:
        List of registered model class names
    """
    return list(_MODEL_CLASSES.keys())


# Initialize registry at module import time (after all functions are defined)
initialize_registry()
