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
from representation_learning.models.probes.minimal_attention_probe import (
    MinimalAttentionProbe,
)
from representation_learning.models.probes.mlp_probe import MLPProbe
from representation_learning.models.probes.transformer_probe import (
    TransformerProbe,
)
from representation_learning.models.probes.weighted_attention_probe import (
    WeightedAttentionProbe,
)
from representation_learning.models.probes.weighted_linear_probe import (
    WeightedLinearProbe,
)
from representation_learning.models.probes.weighted_lstm_probe import (
    WeightedLSTMProbe,
)
from representation_learning.models.probes.weighted_minimal_attention_probe import (
    WeightedMinimalAttentionProbe,
)
from representation_learning.models.probes.weighted_mlp_probe import (
    WeightedMLPProbe,
)
from representation_learning.models.probes.weighted_transformer_probe import (
    WeightedTransformerProbe,
)

logger = logging.getLogger(__name__)


def prune_model_to_layer(
    base_model: ModelBase, frozen: bool, layers: list[str]
) -> ModelBase:
    """
    Prune a base model to keep only layers up to the uppermost specified layer.

    This function:
    1. Validates that the specified layers exist in the base model
    2. Identifies the uppermost layer from the specified layers
    3. Prunes the model to discard all layers above this uppermost layer
    4. Freezes or unfreezes the model based on the frozen parameter

    Args:
        base_model: The base model to prune
        frozen: If True, freeze the pruned model. If False, unfreeze and set to
            train mode
        layers: List of layer names to consider for pruning

    Returns:
        The pruned and configured base model

    Raises:
        ValueError: If none of the specified layers are found in the model
    """
    if not layers:
        raise ValueError("Layers list cannot be empty")

    # Get all named modules to find layer depths
    named_modules = list(base_model.named_modules())

    # Validate that specified layers exist in the model
    existing_layers = [name for name, _ in named_modules]
    found_layers = [layer for layer in layers if layer in existing_layers]

    if not found_layers:
        raise ValueError(
            f"None of the specified layers {layers} were found in the model. "
            f"Available layers: {existing_layers[:10]}..."  # Show first 10 for brevity
        )

    # For the pruning logic, we want to keep only the explicitly requested layers
    # and remove everything else
    layers_to_keep = set(found_layers)

    # Create a new model with only the explicitly requested layers
    pruned_model = _create_pruned_model(base_model, layers_to_keep)

    # Configure freezing/unfreezing
    if frozen:
        pruned_model.eval()
        for param in pruned_model.parameters():
            param.requires_grad = False
    else:
        pruned_model.train()
        for param in pruned_model.parameters():
            param.requires_grad = True

    return pruned_model


def _create_pruned_model(base_model: ModelBase, layers_to_keep: set[str]) -> ModelBase:
    """
    Create a pruned version of the base model keeping only the specified layers.

    Args:
        base_model: The original base model
        layers_to_keep: Set of layer names to keep in the pruned model

    Returns:
        A new pruned model instance
    """
    # Create a new model instance of the same class
    # Try to get constructor arguments from the original model
    constructor_kwargs = {}

    # Add device if available
    if hasattr(base_model, "device"):
        constructor_kwargs["device"] = base_model.device

    # Add audio_config if available
    if hasattr(base_model, "audio_config"):
        constructor_kwargs["audio_config"] = base_model.audio_config

    # Try to create the model with available arguments
    try:
        pruned_model = type(base_model)(**constructor_kwargs)
    except TypeError as e:
        # If constructor fails, try with just device
        if "device" in constructor_kwargs:
            try:
                pruned_model = type(base_model)(device=constructor_kwargs["device"])
            except TypeError:
                # Last resort: try with no arguments
                pruned_model = type(base_model)()
        else:
            raise e

    # Copy the state dict from the original model
    original_state_dict = base_model.state_dict()

    # Filter state dict to keep only parameters from the specified layers
    pruned_state_dict = {}
    for param_name, param_value in original_state_dict.items():
        # Check if this parameter belongs to one of the layers to keep
        if _is_param_in_layers(param_name, layers_to_keep):
            pruned_state_dict[param_name] = param_value

    # Load the filtered state dict into the pruned model
    pruned_model.load_state_dict(pruned_state_dict, strict=False)

    # Actually remove the unwanted layers from the model structure
    _remove_layers_above(pruned_model, layers_to_keep)

    # Copy any additional attributes that might be needed
    for attr_name in ["_layer_names"]:
        if hasattr(base_model, attr_name):
            setattr(pruned_model, attr_name, getattr(base_model, attr_name).copy())

    return pruned_model


def _remove_layers_above(model: ModelBase, layers_to_keep: set[str]) -> None:
    """
    Remove all layers except the specified ones from the model.

    Args:
        model: The model to modify
        layers_to_keep: Set of layer names to keep in the model
    """
    # Get all module names in the model
    module_names = list(model._modules.keys())

    for module_name in module_names:
        # Check if this module should be kept
        if module_name not in layers_to_keep:
            # Remove the module
            delattr(model, module_name)


def _is_param_below_layer(param_name: str, layer_name: str) -> bool:
    """
    Check if a parameter belongs to a layer at or below the specified layer.

    Args:
        param_name: The parameter name (e.g., 'layer1.conv1.weight')
        layer_name: The layer name to check against (e.g., 'layer1.conv1')

    Returns:
        True if the parameter is at or below the specified layer
    """
    # Handle exact match
    if param_name == layer_name:
        return True

    # Handle parameters that belong to the specified layer
    if param_name.startswith(layer_name + "."):
        return True

    # Handle parameters that are at a lower level than the specified layer
    param_parts = param_name.split(".")
    layer_parts = layer_name.split(".")

    # Check if the parameter is at a lower level in the hierarchy
    if len(param_parts) < len(layer_parts):
        return True

    # Check if the parameter is at the same level or lower
    for i, layer_part in enumerate(layer_parts):
        if i >= len(param_parts) or param_parts[i] != layer_part:
            return False

    return True


def _is_param_in_layers(param_name: str, layers_to_keep: set[str]) -> bool:
    """
    Check if a parameter belongs to one of the specified layers.

    Args:
        param_name: The parameter name (e.g., 'layer1.conv1.weight')
        layers_to_keep: Set of layer names to check against

    Returns:
        True if the parameter belongs to one of the specified layers
    """
    # Check if the parameter name starts with any of the layers to keep
    for layer_name in layers_to_keep:
        if param_name == layer_name or param_name.startswith(layer_name + "."):
            return True

    return False


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
    probe_type = probe_config.probe_type.lower()
    layers = probe_config.target_layers
    aggregation = probe_config.aggregation
    input_processing = probe_config.input_processing

    # Log probe creation details
    logger.info(
        f"Creating {probe_type} probe: layers={layers}, "
        f"aggregation={aggregation}, input_processing={input_processing}"
    )

    # Validate input processing compatibility
    if input_processing == "sequence" and probe_type not in [
        "lstm",
        "weighted_lstm",
        "attention",
        "attention_minimal",
        "transformer",
        "weighted_attention",
        "weighted_attention_minimal",
        "weighted_transformer",
    ]:
        raise ValueError(
            f"Sequence input processing is not compatible with {probe_type} probe"
        )

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

    # Use provided target_length if available, otherwise use probe_config.target_length
    final_target_length = (
        target_length if target_length is not None else probe_config.target_length
    )

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
            raise ValueError(
                "LSTM probe requires lstm_hidden_size and num_layers to be specified"
            )

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

    elif probe_type == "weighted_lstm":
        # Extract LSTM-specific parameters
        lstm_hidden_size = probe_config.lstm_hidden_size
        num_layers = probe_config.num_layers
        if lstm_hidden_size is None or num_layers is None:
            raise ValueError(
                "WeightedLSTM probe requires lstm_hidden_size and num_layers "
                "to be specified"
            )

        return WeightedLSTMProbe(
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

    elif probe_type == "weighted_linear":
        return WeightedLinearProbe(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=probe_config.aggregation,
            target_length=final_target_length,
            freeze_backbone=probe_config.freeze_backbone,
        )

    elif probe_type == "weighted_mlp":
        # Extract MLP-specific parameters
        hidden_dims = probe_config.hidden_dims
        if hidden_dims is None:
            raise ValueError("WeightedMLP probe requires hidden_dims to be specified")

        return WeightedMLPProbe(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=probe_config.aggregation,
            hidden_dims=hidden_dims,
            dropout_rate=probe_config.dropout_rate,
            activation=probe_config.activation,
            use_positional_encoding=probe_config.use_positional_encoding,
            target_length=final_target_length,
            projection_dim=probe_config.projection_dim,
            freeze_backbone=probe_config.freeze_backbone,
        )

    elif probe_type == "weighted_attention":
        # Extract attention-specific parameters
        num_heads = probe_config.num_heads
        attention_dim = probe_config.attention_dim
        num_layers = probe_config.num_layers
        if num_heads is None or attention_dim is None or num_layers is None:
            raise ValueError(
                "WeightedAttention probe requires num_heads, attention_dim, and "
                "num_layers to be specified"
            )

        return WeightedAttentionProbe(
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

    elif probe_type == "weighted_attention_minimal":
        return WeightedMinimalAttentionProbe(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=probe_config.aggregation,
            num_heads=probe_config.num_heads or 1,
            target_length=final_target_length,
            freeze_backbone=probe_config.freeze_backbone,
        )

    elif probe_type == "weighted_transformer":
        # Extract transformer-specific parameters
        num_heads = probe_config.num_heads
        attention_dim = probe_config.attention_dim
        num_layers = probe_config.num_layers
        if num_heads is None or attention_dim is None or num_layers is None:
            raise ValueError(
                "WeightedTransformer probe requires num_heads, attention_dim, and "
                "num_layers to be specified"
            )

        return WeightedTransformerProbe(
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

    elif probe_type == "attention":
        # Extract attention-specific parameters
        num_heads = probe_config.num_heads
        attention_dim = probe_config.attention_dim
        num_layers = probe_config.num_layers
        if num_heads is None or attention_dim is None or num_layers is None:
            raise ValueError(
                "Attention probe requires num_heads, attention_dim, and "
                "num_layers to be specified"
            )

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

    elif probe_type == "attention_minimal":
        # Keep the minimal version very lightweight
        return MinimalAttentionProbe(
            base_model=base_model,
            layers=layers,
            num_classes=num_classes,
            device=device,
            feature_mode=feature_mode,
            input_dim=input_dim,
            aggregation=probe_config.aggregation,
            num_heads=probe_config.num_heads or 1,
            target_length=final_target_length,
            freeze_backbone=probe_config.freeze_backbone,
        )

    elif probe_type == "transformer":
        # Extract transformer-specific parameters
        num_heads = probe_config.num_heads
        attention_dim = probe_config.attention_dim
        num_layers = probe_config.num_layers
        if num_heads is None or attention_dim is None or num_layers is None:
            raise ValueError(
                "Transformer probe requires num_heads, attention_dim, and "
                "num_layers to be specified"
            )

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
        supported = (
            "'linear', 'mlp', 'lstm', 'attention', 'attention_minimal', 'transformer', "
            "'weighted_linear', 'weighted_mlp', 'weighted_lstm', 'weighted_attention', "
            "'weighted_attention_minimal', 'weighted_transformer'"
        )
        raise NotImplementedError(
            f"Probe type '{probe_type}' is not implemented. "
            f"Supported types: {supported}"
        )
