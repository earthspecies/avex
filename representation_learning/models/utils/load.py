"""
Model loading utilities for the representation learning framework.

This module provides a simplified interface for loading base models with
minimal configuration, supporting both registered models and external files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch

from representation_learning.configs import ModelSpec
from representation_learning.utils.utils import _process_state_dict

from .factory import build_model_from_spec
from .registry import (
    ensure_initialized,
    get_checkpoint_path,
    get_model,
    get_model_class,
    is_model_class_registered,
    is_registered,
    list_model_names,
    list_models,
    load_model_spec_from_yaml,
    register_model,
)

logger = logging.getLogger(__name__)


def load_model(
    model: Union[str, Path, ModelSpec],
    num_classes: Optional[int] = None,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
    **kwargs: object,
) -> object:
    """Load a complete model (architecture + optionally pre-trained weights).

    **When to use this function:**
    - ✅ Loading pre-trained models with weights
    - ✅ Loading models from checkpoints
    - ✅ Loading models for inference/evaluation
    - ✅ When you need the full loading pipeline

    **When to use build_model() instead:**
    - ❌ Creating new models for training from scratch
    - ❌ When you don't need pre-trained weights
    - ❌ Using custom model classes (use build_model for plugin architecture)

    This function provides a single entry point for loading any model type:

    **Registered Models**: Use predefined model names from the registry
    - `"efficientnet_animalspeak"` - EfficientNet for audio classification
    - `"beats_naturelm"` - BEATs from NatureLM for bioacoustics
    - `"sl_eat_animalspeak_ssl_all"` - EAT for bioacoustics

    **Custom Configs**: Load from YAML configuration files
    - `"path/to/experiment.yml"` - External experiment configs
    - `"configs/run_configs/pretrained/model.yml"` - Local configs

    **Direct Specs**: Use ModelSpec objects directly
    - `ModelSpec(name="efficientnet", ...)` - Programmatic configuration

    Args:
        model: Model identifier, config file path, or ModelSpec object
        num_classes: Number of output classes (optional - extracted from
            checkpoint if available)
        device: Device to load the model on ("cpu", "cuda", etc.)
        checkpoint_path: Optional path to checkpoint weights (supports gs:// paths)
        **kwargs: Additional arguments passed to model creation

    Returns:
        Loaded model ready for training/inference

    Raises:
        ValueError: If model identifier is unknown or invalid
        TypeError: If model type is not supported

    Examples:
        >>> # Load with explicit num_classes (for new model)
        >>> # Note: This requires the model class to be registered first
        >>> # model = load_model("efficientnet_animalspeak", num_classes=100)

        >>> # Load with custom checkpoint
        >>> # model = load_model("efficientnet_animalspeak", checkpoint_path="gs://my-bucket/checkpoint.pt")

        >>> # Load with default checkpoint (if registered)
        >>> # from representation_learning import register_checkpoint
        >>> # register_checkpoint("beats_naturelm", "gs://my-bucket/beats_naturelm.pt")
        >>> # model = load_model("beats_naturelm")  # Uses default checkpoint

        >>> # Load from config file (num_classes from config)
        >>> # model = load_model("experiments/my_model.yml")

        >>> # Load with custom parameters
        >>> # model = load_model("efficientnet_animalspeak", num_classes=50)
    """
    if isinstance(model, str):
        # Ensure registry is initialized
        ensure_initialized()

        # Case 1: Registered model
        if is_registered(model):
            model_spec = get_model(model)
            logger.info(f"Loading registered model: {model}")
            return _load_from_modelspec(
                model_spec,
                num_classes,
                device,
                checkpoint_path,
                registry_key=model,
                **kwargs,
            )

        # Case 2: YAML path
        if model.endswith((".yml", ".yaml")) or Path(model).exists():
            logger.info(f"Loading model from config file: {model}")
            spec = load_model_spec_from_yaml(model)
            # Auto-register the model for future use (by file stem)
            model_name = Path(model).stem
            register_model(model_name, spec)
            return _load_from_modelspec(
                spec,
                num_classes,
                device,
                checkpoint_path,
                registry_key=model_name,
                **kwargs,
            )

        # Case 3: Unknown model identifier
        available_models = list_model_names()
        raise ValueError(
            f"Unknown model identifier: '{model}'. "
            f"Available models: {available_models}. "
            f"Or provide a path to a YAML config file."
        )

    elif isinstance(model, Path):
        # Handle Path objects by converting to string
        return load_model(str(model), num_classes, device, checkpoint_path, **kwargs)

    elif isinstance(model, ModelSpec):
        logger.info("Loading model from ModelSpec object")
        return _load_from_modelspec(
            model, num_classes, device, checkpoint_path, **kwargs
        )

    else:
        raise TypeError(
            f"Unsupported model type: {type(model)}. Expected str, Path, or ModelSpec."
        )


def _load_from_modelspec(
    model_spec: ModelSpec,
    num_classes: Optional[int],
    device: str,
    checkpoint_path: Optional[str],
    registry_key: Optional[str] = None,
    **kwargs: object,
) -> object:
    """Load from ModelSpec object using factory + checkpoint loading.

    Returns:
        object: Loaded model instance.

    Raises:
        ValueError: If model loading fails
    """
    # Override device
    model_spec.device = device

    # Track if num_classes was originally None (means we want to keep classifier)
    num_classes_was_none = num_classes is None

    # Determine checkpoint path with priority:
    # 1. User-provided checkpoint_path parameter (highest priority - already handled)
    # 2. Registry checkpoint path from YAML (second priority)
    if not checkpoint_path and num_classes is None:
        # Use registry key directly if provided (from string identifier or YAML path)
        # Otherwise, find registry key by matching ModelSpec
        if registry_key is not None:
            reg_key = registry_key
        else:
            # Optimized lookup: Filter by model_spec.name first, then compare
            # This avoids comparing against all models when we can filter by name
            all_models = list_models()
            candidates = {
                key: spec
                for key, spec in all_models.items()
                if spec.name == model_spec.name
            }

            # Compare only the filtered candidates (much faster than comparing all)
            reg_key = None
            for key, spec in candidates.items():
                if spec == model_spec:
                    reg_key = key
                    break

        if reg_key is not None:
            default_checkpoint = get_checkpoint_path(reg_key)
            if default_checkpoint:
                checkpoint_path = default_checkpoint
                logger.info(
                    f"Using default checkpoint path from YAML config: {checkpoint_path}"
                )

    # Handle checkpoint loading
    # Note: If a checkpoint_path is provided (either explicitly or from YAML),
    # we set pretrained=False to prevent loading the model's default pretrained weights.
    # This is because:
    # 1. checkpoint_path takes priority over pretrained weights
    #    (user wants specific weights)
    # 2. The checkpoint could be from any source: fine-tuned, SSL-pretrained,
    #    another round of SSL, etc.
    # 3. Setting pretrained=False prevents double-loading of weights
    # The pretrained flag is for the model's built-in default weights
    # (torchvision, HuggingFace, hardcoded paths), not for custom checkpoints from
    # additional training rounds.
    if checkpoint_path:
        model_spec.pretrained = False

    # Extract num_classes from checkpoint if not provided
    if num_classes is None and checkpoint_path:
        num_classes = _extract_num_classes_from_checkpoint(checkpoint_path, device)
        if num_classes is None:
            raise ValueError(
                f"Could not determine num_classes from checkpoint: {checkpoint_path}"
            )
        logger.info(f"Extracted num_classes={num_classes} from checkpoint")
    elif num_classes is None:
        # Try to build model without num_classes for embedding extraction
        # Check if model supports return_features_only to optimize the build
        import inspect

        from .factory import get_model_class

        model_type = model_spec.name
        # Try to check if model supports return_features_only
        # If model class not registered, get_model_class will raise KeyError
        # which will propagate - build_model_from_spec will handle it
        model_class = get_model_class(model_type)
        sig = inspect.signature(model_class.__init__)
        supports_return_features_only = "return_features_only" in sig.parameters

        if supports_return_features_only:
            # Model explicitly supports embedding extraction without classifier
            # Set return_features_only=True and use dummy num_classes
            kwargs["return_features_only"] = True
            # Dummy value (not used when return_features_only=True)
            # This is because some models require num_classes in their signature
            # but it won't be used when return_features_only=True
            # It also doesn't affect models that don't have classifiers (EAT)
            num_classes = 1
            logger.info(
                f"Building {model_type} model without classifier "
                f"(return_features_only=True) for embedding extraction"
            )

    # Create model using factory
    backbone = build_model_from_spec(model_spec, device, num_classes, **kwargs)

    # Load checkpoint if provided
    # Keep classifier weights if num_classes was originally None
    # (extracted from checkpoint)
    if checkpoint_path:
        keep_classifier = num_classes_was_none
        _load_checkpoint(
            backbone, checkpoint_path, device, keep_classifier=keep_classifier
        )

    return backbone.to(device)


def create_model(
    model: Union[str, Path, ModelSpec],
    num_classes: int,
    device: str = "cpu",
    **kwargs: object,
) -> object:
    """Create a new model instance without loading any pre-trained weights.

    **When to use this function:**
    - ✅ Creating new models for training from scratch
    - ✅ When you don't need pre-trained weights
    - ✅ Using custom model classes (plugin architecture)
    - ✅ Building models for fine-tuning

    **When to use load_model() instead:**
    - ❌ Loading pre-trained models with weights
    - ❌ Loading models from checkpoints
    - ❌ Loading models for inference/evaluation

    Args:
        model: Model identifier, config file path, or ModelSpec object
        num_classes: Number of output classes (required)
        device: Device to create the model on ("cpu", "cuda", etc.)
        **kwargs: Additional arguments passed to model creation

    Returns:
        New model instance ready for training

    Raises:
        ValueError: If model identifier is unknown or invalid
        TypeError: If model type is not supported

    Examples:
        >>> # Create new model for training
        >>> # Note: This requires the model class to be registered first
        >>> # model = create_model("efficientnet_animalspeak", num_classes=100)

        >>> # Create custom model using plugin architecture
        >>> # model = create_model("my_custom_model", num_classes=50)

        >>> # Create from config file
        >>> # model = create_model("experiments/my_model.yml", num_classes=10)
    """
    if isinstance(model, str):
        # Ensure registry is initialized
        ensure_initialized()

        # Case 1: Registered model class (plugin architecture)
        if is_model_class_registered(model):
            logger.info(f"Creating model using plugin architecture: {model}")
            model_class = get_model_class(model)
            return model_class(device=device, num_classes=num_classes, **kwargs)

        # Case 2: Registered model spec
        if is_registered(model):
            model_spec = get_model(model)
            logger.info(f"Creating registered model: {model}")
            return build_model_from_spec(model_spec, device, num_classes, **kwargs)

        # Case 3: YAML path
        if model.endswith((".yml", ".yaml")) or Path(model).exists():
            logger.info(f"Creating model from config file: {model}")
            spec = load_model_spec_from_yaml(model)
            # Auto-register the model for future use
            model_name = Path(model).stem
            register_model(model_name, spec)
            return build_model_from_spec(spec, device, num_classes, **kwargs)

        # Case 4: Unknown model identifier
        available_models = list_model_names()
        raise ValueError(
            f"Unknown model identifier: '{model}'. "
            f"Available models: {available_models}. "
            f"Or provide a path to a YAML config file."
        )

    elif isinstance(model, Path):
        # Handle Path objects by converting to string
        return create_model(str(model), num_classes, device, **kwargs)

    elif isinstance(model, ModelSpec):
        logger.info("Creating model from ModelSpec object")
        return build_model_from_spec(model, device, num_classes, **kwargs)

    else:
        raise TypeError(
            f"Unsupported model type: {type(model)}. Expected str, Path, or ModelSpec."
        )


def _extract_num_classes_from_checkpoint(
    checkpoint_path: str, device: str
) -> Optional[int]:
    """Extract num_classes from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        Number of classes if found, None otherwise
    """
    import torch
    from esp_data.io import anypath

    ckpt_path = anypath(checkpoint_path)

    if not ckpt_path.exists():
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return None

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Try to extract from model state dict
        if isinstance(checkpoint, dict):
            # Process state dict to handle prefixes like module. or model.
            # Keep classifier layers since we need them to extract num_classes
            processed_state_dict = _process_state_dict(checkpoint, keep_classifier=True)

            # Look for classifier/head layer weights in processed state dict
            # Prioritize classifier-specific keys
            # (avoid matching backbone layers)
            classifier_keys = [
                key
                for key in processed_state_dict.keys()
                if any(
                    term in key.lower()
                    for term in [
                        "classifier",
                        "head",
                        "classification",
                        "classification_head",
                    ]
                )
                and not any(
                    exclude in key.lower()
                    for exclude in [
                        "backbone",
                        "encoder",
                        "fc1",
                        "fc2",
                        "fc3",
                        "projection",
                        "dense",
                    ]
                )
            ]

            if len(classifier_keys) == 0:
                logger.warning("Could not find classifier/head weights in checkpoint")
                return None

            for key in classifier_keys:
                if "weight" in key and len(processed_state_dict[key].shape) == 2:
                    num_classes = processed_state_dict[key].shape[0]
                    logger.info(f"Found num_classes={num_classes} from {key}")
                    return num_classes
                elif "bias" in key and len(processed_state_dict[key].shape) == 1:
                    num_classes = processed_state_dict[key].shape[0]
                    logger.info(f"Found num_classes={num_classes} from {key}")
                    return num_classes

            # Look for metadata in original checkpoint
            if "num_classes" in checkpoint:
                return checkpoint["num_classes"]
            if (
                "model_config" in checkpoint
                and "num_classes" in checkpoint["model_config"]
            ):
                return checkpoint["model_config"]["num_classes"]

        logger.warning("Could not determine num_classes from checkpoint")
        return None

    except Exception as e:
        logger.warning(f"Error reading checkpoint: {e}")
        return None


def _load_checkpoint(
    model: object, checkpoint_path: str, device: str, keep_classifier: bool = False
) -> None:
    """Load checkpoint weights into model.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        keep_classifier: If True, keep classifier/head layers from checkpoint.
            If False, remove classifier layers (default).

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    from esp_data.io import anypath

    ckpt_path = anypath(checkpoint_path)

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Process state dict if needed
    state_dict = _process_state_dict(checkpoint, keep_classifier=keep_classifier)

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    if keep_classifier:
        logger.info("Checkpoint loaded successfully with classifier/head weights")
    else:
        logger.info("Checkpoint loaded successfully (classifier/head weights removed)")
