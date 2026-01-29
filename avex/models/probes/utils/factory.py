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

from avex.configs import ProbeConfig

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
    num_classes: int,
    device: str,
    base_model: Optional[torch.nn.Module] = None,
    input_dim: Optional[int] = None,
    target_length: Optional[int] = None,
    **kwargs: object,
) -> torch.nn.Module:
    """Build a probe instance from a ProbeConfig.

    This function builds a probe that can operate in two modes:
    - **Online mode** (when `base_model` is provided): Probe is attached to a base model
      for end-to-end training. The frozen state is inferred from `probe_config.freeze_backbone`
      and applied by the probe class during initialization.
    - **Offline mode** (when `input_dim` is provided): Probe operates on pre-computed
      embeddings without requiring a base model. `feature_mode` is automatically set to True.

    Args:
        probe_config: ProbeConfig configuration object
        num_classes: Number of output classes
        device: Device for probe
        base_model: Optional base model to attach probe to (for online mode).
            If provided, probe will be attached for end-to-end training.
        input_dim: Optional input dimension of pre-computed embeddings (for offline mode).
            Required if `base_model` is None.
        target_length: Optional target length in samples. If None, uses `probe_config.target_length`.
        **kwargs: Additional args passed to probe __init__

    Returns:
        Instantiated probe module

    Raises:
        ValueError: If probe configuration is invalid, or if both/neither `base_model`
            and `input_dim` are provided.

    Examples:
        Online mode (with base model):
            >>> from avex import load_model  # doctest: +SKIP
            >>> from avex.configs import ProbeConfig  # doctest: +SKIP
            >>> base = load_model("esp_aves2_naturelm_audio_v1_beats", device="cpu")  # doctest: +SKIP
            >>> cfg = ProbeConfig(probe_type="linear", target_layers=["last_layer"])  # doctest: +SKIP
            >>> probe = build_probe_from_config(cfg, base_model=base, num_classes=50, device="cpu")  # doctest: +SKIP

        Offline mode (with pre-computed embeddings):
            >>> from avex.configs import ProbeConfig  # doctest: +SKIP
            >>> cfg = ProbeConfig(probe_type="mlp", target_layers=["last_layer"])  # doctest: +SKIP
            >>> probe = build_probe_from_config(cfg, input_dim=768, num_classes=50, device="cpu")  # doctest: +SKIP
    """
    # Validate that exactly one of base_model or input_dim is provided
    if base_model is not None and input_dim is not None:
        raise ValueError(
            "Cannot specify both 'base_model' and 'input_dim'. "
            "Use 'base_model' for online mode or 'input_dim' for offline mode."
        )
    if base_model is None and input_dim is None:
        raise ValueError("Must specify either 'base_model' (for online mode) or 'input_dim' (for offline mode).")

    # Determine mode
    feature_mode = base_model is None
    mode_str = "offline" if feature_mode else "online"

    # Validate probe type
    probe_type = probe_config.probe_type.lower()
    probe_class = get_probe_class(probe_type)
    if probe_class is None:
        from .registry import list_probe_classes

        available_classes = list_probe_classes()
        raise ValueError(f"Probe class '{probe_type}' is not registered. Available classes: {available_classes}")

    frozen = probe_config.freeze_backbone if not feature_mode else True
    logger.info(f"Building probe '{probe_type}' with num_classes={num_classes} ({mode_str} mode), frozen={frozen}")
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

    # Register hooks on the base model (only for online mode)
    if not feature_mode and base_model is not None:
        # Note: Freezing/unfreezing is handled by the probe class during initialization
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
        logger.info(f"Successfully built probe '{probe_type}' ({mode_str} mode)")
        return probe
    except Exception as e:
        logger.error(f"Failed to build probe '{probe_type}': {e}")
        raise
