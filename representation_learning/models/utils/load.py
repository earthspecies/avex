"""
Model loading utilities for the representation learning framework.

This module provides a simplified interface for loading base models with
minimal configuration, supporting both registered models and external files.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Optional, Union

from representation_learning.configs import ModelSpec
from representation_learning.io import anypath, exists, filesystem_from_path
from representation_learning.utils.utils import _process_state_dict, universal_torch_load

from . import registry
from .factory import build_model_from_spec
from .registry import (
    get_checkpoint_path,
    get_model_class,
    get_model_spec,
    list_models,
    load_model_spec_from_yaml,
    register_model,
)

logger = logging.getLogger(__name__)


def load_model(
    model: Union[str, Path, ModelSpec],
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
    return_features_only: bool = False,
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
        device: Device to load the model on ("cpu", "cuda", etc.)
        checkpoint_path: Optional path to checkpoint weights (supports gs:// paths)
        return_features_only: If True, force embedding extraction mode when supported

    Returns:
        Loaded model ready for training/inference

    Raises:
        ValueError: If model identifier is unknown or invalid
        TypeError: If model type is not supported

    Examples:
        >>> # Load with default checkpoint (num_classes extracted automatically)
        >>> # model = load_model("efficientnet_animalspeak")

        >>> # Load with custom checkpoint
        >>> # model = load_model("efficientnet_animalspeak", checkpoint_path="gs://my-bucket/checkpoint.pt")

        >>> # Load from config file
        >>> # model = load_model("experiments/my_model.yml")

        >>> # Load for embedding extraction (no classifier)
        >>> # model = load_model("beats_naturelm", return_features_only=True)
    """
    if isinstance(model, str):
        # Case 1: Registered model
        model_spec = get_model_spec(model)
        if model_spec is not None:
            logger.info(f"Loading registered model: {model}")
            return _load_from_modelspec(
                model_spec,
                device=device,
                checkpoint_path=checkpoint_path,
                registry_key=model,
                return_features_only=return_features_only,
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
                device=device,
                checkpoint_path=checkpoint_path,
                registry_key=model_name,
                return_features_only=return_features_only,
            )

        # Case 3: Unknown model identifier
        available_models = list(list_models().keys())
        raise ValueError(
            f"Unknown model identifier: '{model}'. "
            f"Available models: {available_models}. "
            f"Or provide a path to a YAML config file."
        )

    elif isinstance(model, Path):
        # Handle Path objects by converting to string
        return load_model(
            str(model),
            device=device,
            checkpoint_path=checkpoint_path,
            return_features_only=return_features_only,
        )

    elif isinstance(model, ModelSpec):
        logger.info("Loading model from ModelSpec object")
        return _load_from_modelspec(
            model,
            device=device,
            checkpoint_path=checkpoint_path,
            return_features_only=return_features_only,
        )

    else:
        raise TypeError(f"Unsupported model type: {type(model)}. Expected str, Path, or ModelSpec.")


def _load_from_modelspec(
    model_spec: ModelSpec,
    device: str,
    checkpoint_path: Optional[str],
    registry_key: Optional[str] = None,
    return_features_only: bool = False,
) -> object:
    """Load from ModelSpec object using factory + checkpoint loading.

    Returns:
        object: Loaded model instance.

    Raises:
        ValueError: If model loading fails
    """
    # Override device
    model_spec.device = device

    # Determine checkpoint path with priority:
    # 1. User-provided checkpoint_path parameter (highest priority - already handled)
    # 2. Registry checkpoint path from YAML (second priority)
    # Note: We always try to get checkpoint path, even for embedding extraction
    # (we'll just strip the classifier when loading)
    if not checkpoint_path:
        # Use registry key directly if provided (from string identifier or YAML path)
        # Otherwise, find registry key by matching ModelSpec
        if registry_key is not None:
            reg_key = registry_key
        else:
            # Optimized lookup: Filter by model_spec.name first, then compare
            # This avoids comparing against all models when we can filter by name
            all_models = registry._MODEL_REGISTRY
            candidates = {key: spec for key, spec in all_models.items() if spec.name == model_spec.name}

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
                logger.info(f"Using default checkpoint path from YAML config: {checkpoint_path}")

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

    # Determine model type (string) for conditional behavior
    model_type = model_spec.name
    model_class = get_model_class(model_type)
    supports_return_features_only = False
    if model_class is not None:
        sig = inspect.signature(model_class.__init__)
        supports_return_features_only = "return_features_only" in sig.parameters

    # Build internal kwargs dict for model initialization
    # These are runtime-determined values that cannot be in ModelSpec:
    # - num_classes: extracted from checkpoint (varies by checkpoint, not model spec)
    # - return_features_only: user's runtime choice (not static configuration)
    # All other model configuration comes from ModelSpec via build_model_from_spec()
    model_kwargs: dict[str, object] = {}

    # If return_features_only is requested and supported, ensure the flag is set
    if return_features_only and supports_return_features_only:
        model_kwargs["return_features_only"] = True
        logger.info(f"Loading {model_type} model in embedding extraction mode (return_features_only=True)")

    # Extract num_classes from checkpoint if not provided and checkpoint exists
    # This must happen BEFORE building the model so the model is created with the correct classifier
    if checkpoint_path and not return_features_only:
        extracted_num_classes = _extract_num_classes_from_checkpoint(checkpoint_path, device)
        if extracted_num_classes is not None:
            model_kwargs["num_classes"] = extracted_num_classes
            logger.info(f"Extracted num_classes={extracted_num_classes} from checkpoint")
        elif registry_key is not None:
            # Fallback: try to get num_classes from label mapping
            label_mapping = load_label_mapping(registry_key)
            if label_mapping and "label_to_index" in label_mapping:
                num_classes = len(label_mapping["label_to_index"])
                model_kwargs["num_classes"] = num_classes
                logger.info(f"Extracted num_classes={num_classes} from label mapping")
        else:
            # Checkpoint exists but no num_classes found - likely a backbone-only checkpoint
            # Automatically enable embedding mode for models that support it
            if supports_return_features_only:
                return_features_only = True
                model_kwargs["return_features_only"] = True
                logger.info(
                    f"Checkpoint found but no classifier detected; loading {model_type} in embedding extraction mode"
                )

    # If pretrained=True, pretrained weights are typically backbone-only (no classifier)
    # Automatically enable embedding mode for models that support it
    if model_spec.pretrained and not checkpoint_path and supports_return_features_only and not return_features_only:
        return_features_only = True
        model_kwargs["return_features_only"] = True
        logger.info(
            f"Model '{registry_key or model_type}' has pretrained=True (backbone-only); "
            "automatically enabling embedding extraction mode"
        )

    # If no checkpoint and not in embedding mode and not pretrained, we no longer support creating
    # new classifier heads via load_model. Use backbones + probes instead.
    # Exception: If the model supports return_features_only, let it handle num_classes=None
    # (e.g., BEATs automatically sets return_features_only=True when num_classes=None)
    if (
        not checkpoint_path
        and not return_features_only
        and not model_spec.pretrained
        and not supports_return_features_only
    ):
        raise ValueError(
            "load_model() without a checkpoint no longer creates new classifier heads. "
            "Build a backbone with build_model()/build_model_from_spec() and attach "
            "a probe head via build_probe_from_config() instead."
        )

    # Create model using factory (backbone; classifier, if any, is defined by the class or checkpoint)
    # ModelSpec contains all static configuration; model_kwargs only contains runtime-determined values
    # (num_classes from checkpoint, return_features_only from user choice - these cannot be in ModelSpec)
    backbone = build_model_from_spec(model_spec, device, **model_kwargs)

    # Load label mapping if available (only for models with classifier heads)
    # Don't load label mapping if return_features_only=True
    if not return_features_only and registry_key is not None:
        label_mapping = load_label_mapping(registry_key)
        if label_mapping:
            # Attach label mapping to model for easy access
            backbone.label_mapping = label_mapping
            logger.info(
                f"Attached label mapping to model (label_to_index: {len(label_mapping['label_to_index'])} classes)"
            )

    # Load checkpoint if provided
    if checkpoint_path:
        if return_features_only:
            # For embedding extraction, always strip the classifier from checkpoint
            keep_classifier = False
            logger.info("Loading checkpoint for embedding extraction (classifier will be stripped)")
        else:
            # For classification mode, always keep whatever classifier is in the checkpoint
            keep_classifier = True
            logger.info("Loading checkpoint and keeping classifier head from checkpoint when present")
        _load_checkpoint(backbone, checkpoint_path, device, keep_classifier=keep_classifier)

    return backbone.to(device)


def _extract_num_classes_from_checkpoint(checkpoint_path: str, device: str) -> Optional[int]:
    """Extract num_classes from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        Number of classes if found, None otherwise
    """
    ckpt_path = anypath(checkpoint_path)

    if not exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return None

    try:
        checkpoint = universal_torch_load(ckpt_path, map_location=device)

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

            # Find the final output layer (typically the last classifier layer with smallest output dim)
            # Sort keys to process in order, and prefer the last one (final output layer)
            classifier_keys_sorted = sorted(classifier_keys)
            for key in reversed(classifier_keys_sorted):  # Process from last to first
                if "weight" in key and len(processed_state_dict[key].shape) == 2:
                    # For 2D weight matrices, the first dimension is typically the output size
                    # But for the final layer, we want the smallest output dimension
                    output_dim = processed_state_dict[key].shape[0]
                    # Heuristic: final output layer usually has output_dim < 1000 (reasonable num_classes)
                    # and is one of the last layers
                    if output_dim < 10000:  # Reasonable upper bound for num_classes
                        num_classes = output_dim
                        logger.info(f"Found num_classes={num_classes} from {key}")
                        return num_classes
                elif "bias" in key and len(processed_state_dict[key].shape) == 1:
                    # For bias vectors, the length is the output size
                    output_dim = processed_state_dict[key].shape[0]
                    if output_dim < 10000:  # Reasonable upper bound for num_classes
                        num_classes = output_dim
                        logger.info(f"Found num_classes={num_classes} from {key}")
                        return num_classes

            # Look for metadata in original checkpoint
            if "num_classes" in checkpoint:
                return checkpoint["num_classes"]
            if "model_config" in checkpoint and "num_classes" in checkpoint["model_config"]:
                return checkpoint["model_config"]["num_classes"]

        logger.warning("Could not determine num_classes from checkpoint")
        return None

    except Exception as e:
        logger.warning(f"Error reading checkpoint: {e}")
        return None


def load_label_mapping(model_or_path: Union[str, Path]) -> Optional[dict]:
    """Load label mapping from JSON file.

    The label mapping is a JSON dictionary that maps class labels to their
    corresponding indices (logit positions) in the classifier head.

    Args:
        model_or_path: Either a model name (str) to load mapping from YAML config,
            or a path (str/Path) to the JSON file containing the label mapping.
            Supports cloud storage paths (e.g., gs://).

    Returns:
        Dictionary with 'label_to_index' and 'index_to_label' mappings, or None
        if file not found or invalid.

    Example:
        >>> # Load by model name (reads path from YAML config)
        >>> mapping = load_label_mapping("sl_beats_animalspeak")
        >>>
        >>> # Load by direct path
        >>> mapping = load_label_mapping("gs://bucket/label_map.json")
        >>> # mapping = {"label_to_index": {...}, "index_to_label": {...}}
    """
    import json

    # If it's a model name, get the path from YAML config
    if (
        isinstance(model_or_path, str)
        and not any(model_or_path.startswith(prefix) for prefix in ("gs://", "s3://", "http://", "https://", "/"))
        and Path(model_or_path).suffix != ".json"
    ):
        # Likely a model name - get path from YAML config
        from importlib import resources

        from .registry import _MODEL_REGISTRY, _OFFICIAL_MODELS_PKG

        if model_or_path not in _MODEL_REGISTRY:
            logger.warning(f"Model '{model_or_path}' is not registered")
            return None

        # Read class_mapping_path from YAML file
        try:
            root = resources.files(_OFFICIAL_MODELS_PKG)
            yaml_file = root / f"{model_or_path}.yml"
            if yaml_file.is_file():
                with yaml_file.open("r", encoding="utf-8") as f:
                    import yaml

                    yaml_data = yaml.safe_load(f)
                if isinstance(yaml_data, dict) and "class_mapping_path" in yaml_data:
                    class_mapping_path = yaml_data["class_mapping_path"]
                    if class_mapping_path:
                        mapping_path_str = class_mapping_path
                    else:
                        logger.warning(f"No class mapping path found for model: {model_or_path}")
                        return None
                else:
                    logger.warning(f"No class_mapping_path in YAML for model: {model_or_path}")
                    return None
            else:
                logger.warning(f"No YAML file found for model: {model_or_path}")
                return None
        except Exception as e:
            logger.debug(f"Failed to read class_mapping_path from YAML for {model_or_path}: {e}")
            return None
    else:
        # It's a path
        mapping_path_str = str(model_or_path)

    try:
        mapping_path = anypath(mapping_path_str)
        if not exists(mapping_path):
            logger.warning(f"Class mapping file not found: {mapping_path_str}")
            return None

        fs = filesystem_from_path(mapping_path)
        with fs.open(str(mapping_path), mode="r", encoding="utf-8") as f:
            mapping = json.load(f)

        if not isinstance(mapping, dict):
            logger.warning(f"Class mapping file does not contain a dictionary: {mapping_path_str}")
            return None

        # Create reverse mapping (index -> label) for easier lookup
        # The original mapping is label -> index, but we also want index -> label
        reverse_mapping = {v: k for k, v in mapping.items()}
        logger.info(f"Loaded class mapping with {len(mapping)} classes from {mapping_path_str}")

        # Return both mappings in a structured format
        return {
            "label_to_index": mapping,
            "index_to_label": reverse_mapping,
        }
    except Exception as e:
        logger.warning(f"Error loading class mapping from {mapping_path_str}: {e}")
        return None


def _load_checkpoint(model: object, checkpoint_path: str, device: str, keep_classifier: bool = False) -> None:
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
    ckpt_path = anypath(checkpoint_path)

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    if not exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint
    checkpoint = universal_torch_load(ckpt_path, map_location=device)

    # Process state dict if needed
    state_dict = _process_state_dict(checkpoint, keep_classifier=keep_classifier)

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    if keep_classifier:
        logger.info("Checkpoint loaded successfully with classifier/head weights")
    else:
        logger.info("Checkpoint loaded successfully (classifier/head weights removed)")
