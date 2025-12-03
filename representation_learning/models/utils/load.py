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
        # Case 1: Registered model
        model_spec = get_model_spec(model)
        if model_spec is not None:
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
        available_models = list(list_models().keys())
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
        return _load_from_modelspec(model, num_classes, device, checkpoint_path, **kwargs)

    else:
        raise TypeError(f"Unsupported model type: {type(model)}. Expected str, Path, or ModelSpec.")


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

    # Check if return_features_only is explicitly requested
    return_features_only = kwargs.get("return_features_only", False)

    # Track if num_classes was originally None (means we want to keep classifier)
    # But only if return_features_only is False
    num_classes_was_none = num_classes is None and not return_features_only

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

    # Determine return_features_only and num_classes logic
    model_type = model_spec.name
    model_class = get_model_class(model_type)
    supports_return_features_only = False
    if model_class is not None:
        sig = inspect.signature(model_class.__init__)
        supports_return_features_only = "return_features_only" in sig.parameters

    # Logic for return_features_only and num_classes:
    # 1. If return_features_only=True (explicit or inferred): strip classifier, return embeddings
    # 2. If num_classes=None with checkpoint: extract classes, keep classifier
    # 3. If num_classes is explicit: create new classifier
    # 4. If num_classes=None without checkpoint: try embedding extraction if supported

    if return_features_only:
        # Explicit embedding extraction mode
        # Always strip classifier from checkpoint, return embeddings
        kwargs["return_features_only"] = True
        num_classes = None  # Not needed for embedding extraction
        logger.info(f"Building {model_type} model for embedding extraction (return_features_only=True)")
    elif num_classes is None and checkpoint_path:
        # Extract num_classes from checkpoint and keep the classifier
        num_classes = _extract_num_classes_from_checkpoint(checkpoint_path, device)
        if num_classes is None:
            raise ValueError(f"Could not determine num_classes from checkpoint: {checkpoint_path}")
        logger.info(f"Extracted num_classes={num_classes} from checkpoint")
        # Don't set return_features_only - we want the classifier
    elif num_classes is None:
        # No checkpoint, no explicit num_classes
        # Check if model has pretrained=True (checkpoint embedded in model class)
        if model_spec.pretrained:
            # Model has pretrained weights embedded (e.g., BEATs with hardcoded pretrained paths)
            # Pretrained models typically don't have classifier heads, so use return_features_only
            if supports_return_features_only:
                kwargs["return_features_only"] = True
                num_classes = None
                logger.info(
                    f"Building {model_type} model with pretrained=True (embedded checkpoint). "
                    f"Using return_features_only=True for embedding extraction."
                )
            else:
                # Model has pretrained weights but doesn't support return_features_only
                # This shouldn't happen for most models, but handle gracefully
                raise ValueError(
                    f"num_classes must be provided for {model_type} model "
                    f"(model has pretrained=True but doesn't support return_features_only=True)"
                )
        elif supports_return_features_only:
            # No pretrained weights, model supports embedding extraction
            kwargs["return_features_only"] = True
            num_classes = None
            logger.info(
                f"Building {model_type} model without classifier (return_features_only=True) for embedding extraction"
            )
        else:
            # Model doesn't support return_features_only and no pretrained weights, need num_classes
            raise ValueError(
                f"num_classes must be provided for {model_type} model "
                f"(model does not support return_features_only=True and pretrained=False)"
            )
    # else: num_classes is explicitly provided - create new classifier

    # Create model using factory
    backbone = build_model_from_spec(model_spec, device, num_classes, **kwargs)

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
    # When return_features_only=True: always strip classifier (keep_classifier=False)
    # When return_features_only=False and num_classes was None: keep classifier (keep_classifier=True)
    # When return_features_only=False and num_classes is explicit: strip classifier (keep_classifier=False)
    if checkpoint_path:
        if return_features_only:
            # For embedding extraction, always strip the classifier from checkpoint
            keep_classifier = False
            logger.info("Loading checkpoint for embedding extraction (classifier will be stripped)")
        else:
            # For classification mode, keep classifier only if num_classes was originally None
            keep_classifier = num_classes_was_none
        _load_checkpoint(backbone, checkpoint_path, device, keep_classifier=keep_classifier)

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
        # Case 1: Registered model class (plugin architecture)
        model_class = get_model_class(model)
        if model_class is not None:
            logger.info(f"Creating model using plugin architecture: {model}")
            return model_class(device=device, num_classes=num_classes, **kwargs)

        # Case 2: Registered model spec
        model_spec = get_model_spec(model)
        if model_spec is not None:
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
        available_models = list(list_models().keys())
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
        raise TypeError(f"Unsupported model type: {type(model)}. Expected str, Path, or ModelSpec.")


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
