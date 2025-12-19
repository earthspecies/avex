"""
Probe factory for building probe instances from registered classes and ProbeConfig.

This module provides functionality to build probe instances by linking
ProbeConfig configurations with registered probe classes. Mirrors the
model factory structure.
"""

from __future__ import annotations

import inspect
import logging
from typing import Optional

import torch

from representation_learning.configs import ProbeConfig

from .registry import get_probe_class

logger = logging.getLogger(__name__)


def _add_probe_config_params(init_kwargs: dict, probe_config: ProbeConfig) -> None:
    """Add probe-specific parameters from ProbeConfig to init_kwargs.

    This function dynamically extracts all non-None parameters from ProbeConfig
    and adds them to the initialization kwargs, avoiding repetitive manual listing.

    Args:
        init_kwargs: Dictionary to add parameters to
        probe_config: ProbeConfig object to extract parameters from
    """
    # Define parameters that should be included if they exist and are not None
    param_names = [
        "hidden_dims",
        "dropout_rate",
        "activation",
        "lstm_hidden_size",
        "num_layers",
        "bidirectional",
        "max_sequence_length",
        "use_positional_encoding",
        "num_heads",
        "attention_dim",
    ]

    for param_name in param_names:
        if hasattr(probe_config, param_name):
            value = getattr(probe_config, param_name)
            # Only add if value is not None and not empty string
            if value is not None and value != "":
                init_kwargs[param_name] = value


def build_probe_from_config(
    probe_config: ProbeConfig,
    base_model: Optional[torch.nn.Module],
    num_classes: int,
    device: str,
    feature_mode: bool = False,
    input_dim: Optional[int] = None,
    frozen: bool = True,
    target_length: Optional[int] = None,
    **kwargs: object,
) -> torch.nn.Module:
    """
    Build a probe instance directly from a ProbeConfig object.

    This function directly instantiates probes using the registry pattern,
    similar to build_model_from_spec.

    Args:
        probe_config: ProbeConfig configuration object
        base_model: Base model to attach probe to (can be None for offline mode)
        num_classes: Number of output classes
        device: Device for probe
        feature_mode: Whether to use input directly as embeddings (offline mode)
        input_dim: Input dimension when in feature mode (required if base_model is None)
        frozen: Whether to freeze the base model
        target_length: Optional target length in samples
        **kwargs: Additional args passed to probe __init__

    Returns:
        Instantiated probe module

    Raises:
        ValueError: If probe configuration is invalid

    Example:
        >>> from representation_learning.configs import ProbeConfig
        >>> # Offline mode: build probe without base model (for pre-computed embeddings)
        >>> probe_config = ProbeConfig(
        ...     probe_type="linear",
        ...     target_layers=["last_layer"],
        ...     aggregation="mean"
        ... )
        >>> probe = build_probe_from_config(
        ...     probe_config,
        ...     base_model=None,
        ...     num_classes=50,
        ...     device="cpu",
        ...     feature_mode=True,
        ...     input_dim=768
        ... )
    """
    # Validate probe type
    probe_type = probe_config.probe_type.lower()
    probe_class = get_probe_class(probe_type)
    if probe_class is None:
        from .registry import list_probe_classes

        available_classes = list_probe_classes()
        raise ValueError(f"Probe class '{probe_type}' is not registered. Available classes: {available_classes}")

    logger.info(f"Building probe '{probe_type}' with num_classes={num_classes}, frozen={frozen}")
    logger.debug(f"Probe config: {probe_config.model_dump()}")

    # Extract common parameters
    layers = probe_config.target_layers
    aggregation = probe_config.aggregation
    input_processing = probe_config.input_processing

    # Validate input processing compatibility
    if input_processing == "sequence" and probe_type not in [
        "lstm",
        "attention",
        "transformer",
    ]:
        raise ValueError(f"Sequence input processing is not compatible with {probe_type} probe")

    # Handle base model freezing/unfreezing
    if not frozen and base_model is not None:
        # Enable training mode
        base_model.train()
        for p in base_model.parameters():
            p.requires_grad = True
    elif frozen and base_model is not None:
        # Freeze the base model
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad = False

    # Register hooks AFTER setting model mode to ensure they work correctly
    if base_model is not None and not feature_mode:
        if hasattr(base_model, "register_hooks_for_layers"):
            layers = base_model.register_hooks_for_layers(layers)
        else:
            logger.warning("base_model does not have register_hooks_for_layers method")

    # Use provided target_length if available, otherwise use probe_config.target_length
    final_target_length = target_length if target_length is not None else probe_config.target_length

    # Prepare initialization arguments with common parameters
    init_kwargs = {
        "base_model": base_model,
        "layers": layers,
        "num_classes": num_classes,
        "device": device,
        "feature_mode": feature_mode,
        "input_dim": input_dim,
        "aggregation": aggregation,
        "target_length": final_target_length,
        "freeze_backbone": frozen,
        **kwargs,
    }

    # Add probe-specific parameters from ProbeConfig dynamically
    _add_probe_config_params(init_kwargs, probe_config)

    # Filter init_kwargs to only include parameters accepted by the probe class
    sig = inspect.signature(probe_class.__init__)
    valid_params = set(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in init_kwargs.items() if k in valid_params}

    logger.debug(f"Filtered initialization kwargs: {filtered_kwargs}")

    # Instantiate the probe
    try:
        probe = probe_class(**filtered_kwargs)
        logger.info(f"Successfully built probe '{probe_type}'")
        return probe
    except Exception as e:
        logger.error(f"Failed to build probe '{probe_type}': {e}")
        raise


def build_probe_from_config_online(
    probe_config: ProbeConfig,
    base_model: torch.nn.Module,
    num_classes: int,
    device: str,
    target_length: Optional[int] = None,
    **kwargs: object,
) -> torch.nn.Module:
    """Build a probe instance for online training (attached to a base model).

    This function builds a probe that is attached to a base model for end-to-end
    training. The frozen state is inferred from probe_config.freeze_backbone.

    Args:
        probe_config: ProbeConfig configuration object
        base_model: Base model to attach probe to
        num_classes: Number of output classes
        device: Device for probe
        target_length: Optional target length in samples
        **kwargs: Additional args passed to probe __init__

    Returns:
        Instantiated probe module attached to base_model
    """
    frozen = probe_config.freeze_backbone
    return build_probe_from_config(
        probe_config=probe_config,
        base_model=base_model,
        num_classes=num_classes,
        device=device,
        feature_mode=False,
        input_dim=None,
        frozen=frozen,
        target_length=target_length,
        **kwargs,
    )


def build_probe_from_config_offline(
    probe_config: ProbeConfig,
    input_dim: int,
    num_classes: int,
    device: str,
    target_length: Optional[int] = None,
    **kwargs: object,
) -> torch.nn.Module:
    """Build a probe instance for offline training (on pre-computed embeddings).

    This function builds a probe that operates on pre-computed embeddings,
    without requiring a base model. feature_mode is automatically set to True.

    Args:
        probe_config: ProbeConfig configuration object
        input_dim: Input dimension of pre-computed embeddings
        num_classes: Number of output classes
        device: Device for probe
        target_length: Optional target length in samples
        **kwargs: Additional args passed to probe __init__

    Returns:
        Instantiated probe module for offline training
    """
    return build_probe_from_config(
        probe_config=probe_config,
        base_model=None,
        num_classes=num_classes,
        device=device,
        feature_mode=True,
        input_dim=input_dim,
        frozen=True,  # Not applicable for offline mode, but required by build_probe_from_config
        target_length=target_length,
        **kwargs,
    )
