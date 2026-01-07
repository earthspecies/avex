"""Factory function to obtain probe instances based on configuration."""

import logging
from typing import Optional

import torch

from representation_learning.configs import ProbeConfig
from representation_learning.models.base_model import ModelBase
from representation_learning.models.probes.attention_probe import (
    AttentionProbe,
)
from representation_learning.models.probes.linear_probe import LinearProbe
from representation_learning.models.probes.lstm_probe import LSTMProbe
from representation_learning.models.probes.mlp_probe import MLPProbe
from representation_learning.models.probes.transformer_probe import TransformerProbe

logger = logging.getLogger(__name__)


def get_probe(
    probe_config: ProbeConfig,
    base_model: Optional[ModelBase],
    num_classes: int,
    device: str = "cuda",
    feature_mode: bool = False,
    input_dim: Optional[int] = None,
    frozen: bool = False,
    target_length: Optional[int] = None,
) -> torch.nn.Module:
    """
    Factory function to obtain a probe instance based on configuration.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Please use :func:`build_probe_from_config` from
        ``representation_learning.models.probes.utils.factory`` instead,
        which provide a registry-based, more maintainable approach.

    This function creates the appropriate probe type based on the probe_config
    and returns an initialized probe model.

    Args:
        probe_config: Probe configuration object containing:
            - probe_type: Type of probe to instantiate
            - target_layers: List of layer names to extract embeddings from
            - aggregation: How to aggregate multiple layer embeddings
            - input_processing: How to process input embeddings
            - Various probe-specific parameters (hidden_dims, num_heads, etc.)
        base_model: Frozen backbone network to pull embeddings from. Can be None if
            feature_mode=True.
        num_classes: Number of output classes.
        device: Device to run on.
        feature_mode: Whether to use the input directly as embeddings.
        input_dim: Input dimension when in feature mode. Required if base_model is None.
        target_length: Optional target length in samples. If provided, overrides
            probe_config.target_length. If None, uses probe_config.target_length.

    Returns:
        An instance of the corresponding probe configured with the provided
        parameters.

    Raises:
        NotImplementedError: If the probe_type is not supported.
        ValueError: If required parameters are missing for the probe type.
    """
    logger.warning(
        "get_probe() is deprecated and will be removed in a future version. "
        "Use build_probe_from_config() "
        "from representation_learning.models.probes.utils.factory instead."
    )

    probe_type = probe_config.probe_type.lower()
    layers = probe_config.target_layers
    aggregation = probe_config.aggregation
    input_processing = probe_config.input_processing

    # Log probe creation details
    logger.info(
        f"Creating {probe_type} probe: layers={layers}, aggregation={aggregation}, input_processing={input_processing}"
    )

    # Validate input processing compatibility
    if input_processing == "sequence" and probe_type not in [
        "lstm",
        "attention",
        "transformer",
    ]:
        raise ValueError(f"Sequence input processing is not compatible with {probe_type} probe")

    # Register hooks on the base model
    # Note: Freezing/unfreezing is handled by the probe class during initialization
    if base_model is not None and not feature_mode:
        layers = base_model.register_hooks_for_layers(layers)

    # Use provided target_length if available, otherwise use probe_config.target_length
    final_target_length = target_length if target_length is not None else probe_config.target_length

    if probe_type == "linear":
        return LinearProbe(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=aggregation,
            target_length=final_target_length,
            freeze_backbone=frozen,
        )

    elif probe_type == "mlp":
        # Extract MLP-specific parameters
        hidden_dims = probe_config.hidden_dims
        if hidden_dims is None:
            raise ValueError("MLP probe requires hidden_dims to be specified")

        return MLPProbe(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=aggregation,
            hidden_dims=hidden_dims,
            dropout_rate=probe_config.dropout_rate,
            activation=probe_config.activation,
            target_length=final_target_length,
        )

    elif probe_type == "lstm":
        # Extract LSTM-specific parameters
        lstm_hidden_size = probe_config.lstm_hidden_size
        num_layers = probe_config.num_layers
        if lstm_hidden_size is None or num_layers is None:
            raise ValueError("LSTM probe requires lstm_hidden_size and num_layers to be specified")

        return LSTMProbe(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=probe_config.aggregation,
            lstm_hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            bidirectional=probe_config.bidirectional,
            dropout_rate=probe_config.dropout_rate,
            max_sequence_length=probe_config.max_sequence_length,
            use_positional_encoding=probe_config.use_positional_encoding,
            target_length=final_target_length,
            freeze_backbone=probe_config.freeze_backbone,
        )

    elif probe_type == "attention":
        # Extract attention-specific parameters
        num_heads = probe_config.num_heads
        attention_dim = probe_config.attention_dim
        num_layers = probe_config.num_layers
        if num_heads is None or attention_dim is None or num_layers is None:
            raise ValueError("Attention probe requires num_heads, attention_dim, and num_layers to be specified")

        return AttentionProbe(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=probe_config.aggregation,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
            dropout_rate=probe_config.dropout_rate,
            max_sequence_length=probe_config.max_sequence_length,
            use_positional_encoding=probe_config.use_positional_encoding,
            target_length=final_target_length,
            freeze_backbone=probe_config.freeze_backbone,
        )

    elif probe_type == "transformer":
        # Extract transformer-specific parameters
        num_heads = probe_config.num_heads
        attention_dim = probe_config.attention_dim
        num_layers = probe_config.num_layers
        if num_heads is None or attention_dim is None or num_layers is None:
            raise ValueError("Transformer probe requires num_heads, attention_dim, and num_layers to be specified")

        return TransformerProbe(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=probe_config.aggregation,
            num_heads=num_heads,
            attention_dim=attention_dim,
            num_layers=num_layers,
            dropout_rate=probe_config.dropout_rate,
            max_sequence_length=probe_config.max_sequence_length,
            use_positional_encoding=probe_config.use_positional_encoding,
            target_length=final_target_length,
            freeze_backbone=probe_config.freeze_backbone,
        )

    else:
        supported = "'linear', 'mlp', 'lstm', 'attention', transformer'"
        raise NotImplementedError(f"Probe type '{probe_type}' is not implemented. Supported types: {supported}")
