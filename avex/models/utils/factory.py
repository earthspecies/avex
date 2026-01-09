"""
Model factory for building model instances from registered classes and ModelSpec.

This module provides functionality to build model instances by linking
ModelSpec configurations with registered model classes.
"""

import inspect
import logging

from avex.configs import AudioConfig, ModelSpec
from avex.models.base_model import ModelBase

from .registry import get_model_class, get_model_spec, list_model_classes, list_models

logger = logging.getLogger(__name__)


def _add_model_spec_params(init_kwargs: dict, model_spec: ModelSpec) -> None:
    """Add model-specific parameters from ModelSpec to init_kwargs.

    This function dynamically extracts all non-None parameters from ModelSpec
    and adds them to the initialization kwargs, avoiding repetitive if statements.

    Args:
        init_kwargs: Dictionary to add parameters to
        model_spec: ModelSpec object to extract parameters from
    """
    # Define parameters that should be included if they exist and are not None
    param_names = [
        "text_model_name",
        "projection_dim",
        "temperature",
        "eat_cfg",
        "pretraining_mode",
        "handle_padding",
        "fairseq_weights_path",
        "eat_norm_mean",
        "eat_norm_std",
        "efficientnet_variant",
        "use_naturelm",
        "fine_tuned",
        "language",
        "model_id",
    ]

    for param_name in param_names:
        if hasattr(model_spec, param_name):
            value = getattr(model_spec, param_name)
            # Only add if value is not None and not empty string
            if value is not None and value != "":
                init_kwargs[param_name] = value


def build_model(model_name: str, device: str, **kwargs: object) -> ModelBase:
    """
    Build a model instance from a registered model class and ModelSpec.

    Args:
        model_name: Name of the model in the registry (ModelSpec)
        device: Device for model
        **kwargs: Additional args passed to model __init__

    Returns:
        Instantiated model class

    Raises:
        ValueError: If ModelSpec not found
        KeyError: If model class not registered
    """
    # Get the ModelSpec for this model
    model_spec = get_model_spec(model_name)
    if model_spec is None:
        available_models = list(list_models().keys())
        raise ValueError(f"No ModelSpec found for '{model_name}'. Available models: {available_models}")

    # Get the model class using the model type from ModelSpec
    model_type = model_spec.name  # e.g., 'beats', 'efficientnet', etc.
    model_class = get_model_class(model_type)
    if model_class is None:
        available_classes = list_model_classes()
        raise KeyError(f"Model class '{model_type}' is not registered. Available classes: {available_classes}")

    # Prepare initialization arguments
    init_kwargs = {
        "device": device,
        "audio_config": model_spec.audio_config.model_dump() if model_spec.audio_config else None,
        **kwargs,
    }

    # Add model-specific parameters from ModelSpec dynamically
    _add_model_spec_params(init_kwargs, model_spec)

    logger.info(f"Building model '{model_name}' with class '{model_type}'")
    logger.debug(f"Initialization kwargs: {init_kwargs}")

    # Instantiate the model
    try:
        model = model_class(**init_kwargs)
        logger.info(f"Successfully built model '{model_name}'")
        return model
    except Exception as e:
        logger.error(f"Failed to build model '{model_name}': {e}")
        raise


def build_model_from_spec(model_spec: ModelSpec, device: str, **kwargs: object) -> ModelBase:
    """
    Build a model instance directly from a ModelSpec object.

    Args:
        model_spec: ModelSpec configuration object
        device: Device for model
        **kwargs: Additional args passed to model __init__

    Returns:
        Instantiated model class

    Raises:
        KeyError: If model class is not registered
    """
    # Get the model class using the model type from ModelSpec
    model_type = model_spec.name
    model_class = get_model_class(model_type)
    if model_class is None:
        available_classes = list_model_classes()
        raise KeyError(f"Model class '{model_type}' is not registered. Available classes: {available_classes}")

    # Prepare initialization arguments (same logic as build_model)
    # Convert audio_config dict to AudioConfig object if needed
    audio_config = None
    if model_spec.audio_config:
        if isinstance(model_spec.audio_config, AudioConfig):
            audio_config = model_spec.audio_config
        else:
            # If it's a dict, convert to AudioConfig
            audio_config = AudioConfig(**model_spec.audio_config)

    init_kwargs = {
        "device": device,
        "audio_config": audio_config,
        **kwargs,
    }

    # Add model-specific parameters from ModelSpec dynamically
    _add_model_spec_params(init_kwargs, model_spec)

    # Filter init_kwargs to only include parameters accepted by the model class
    # This prevents passing model-specific parameters (e.g., eat_norm_mean) to
    # models that don't accept them (e.g., BEATs)
    sig = inspect.signature(model_class.__init__)
    valid_params = set(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in init_kwargs.items() if k in valid_params}

    logger.info(f"Building model from spec with class '{model_type}'")
    logger.debug(f"Initialization kwargs: {filtered_kwargs}")

    # Instantiate the model
    try:
        model = model_class(**filtered_kwargs)
        logger.info("Successfully built model from spec")
        return model
    except Exception as e:
        logger.error(f"Failed to build model from spec: {e}")
        raise
