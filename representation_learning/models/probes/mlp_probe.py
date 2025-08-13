"""MLP probe for flexible probing system."""

from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.models.probes.base_probe import BaseProbe


class MLPProbe(BaseProbe):
    """MLP probe for classification tasks.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension (number of classes)
        base_model: Optional base model for feature extraction
        feature_mode: Whether to use the input directly as embeddings.
        input_dim: Input dimension when in feature mode. Required if base_model is None.
        aggregation: How to aggregate multiple layer embeddings ('mean', 'max',
                    'concat').
        hidden_dims: List of hidden layer dimensions.
        dropout_rate: Dropout rate for regularization.
        activation: Activation function to use.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_model: Optional[nn.Module] = None,
        feature_mode: bool = False,
        aggregation: str = "mean",
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        use_positional_encoding: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            base_model=base_model,
            feature_mode=feature_mode,
            aggregation=aggregation,
        )

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Determine classifier input dimension
        if self.feature_mode:
            # Embeddings will be fed directly – base_model may be None.
            inferred_dim = input_dim
        else:
            # We will compute embeddings inside forward – need base_model.
            if base_model is None:
                raise ValueError("base_model must be provided when feature_mode=False")
            # For now, use a default dimension - this should be updated based on
            # actual usage
            inferred_dim = input_dim

        # Build MLP layers
        self.mlp = self._build_mlp(inferred_dim, output_dim)

    def _build_mlp(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build the MLP architecture.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension

        Returns:
            MLP module
        """
        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    self._get_activation(self.activation),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module.

        Args:
            activation: Activation function name

        Returns:
            Activation function module

        Raises:
            ValueError: If activation function is not supported
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "swish":
            return nn.SiLU()  # SiLU is the same as Swish
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the MLP probe.

        Args:
            x: Input tensor of shape (batch_size, time_steps)
                or (batch_size, embedding_dim) in feature mode
            padding_mask: Optional padding mask tensor of shape (batch_size, time_steps)
        Returns:
            Classification logits of shape (batch_size, num_classes)
        Raises:
            ValueError: If base_model is None when not in feature mode
        """
        if self.feature_mode:
            embeddings = x  # type: ignore[arg-type]
        else:
            if self.base_model is None:
                raise ValueError("base_model must be provided when not in feature mode")
            embeddings = self.base_model.extract_embeddings(
                x, self.layers, padding_mask=padding_mask
            )

        # Apply aggregation if multiple layers
        if len(self.layers) > 1:
            embeddings = self._aggregate_embeddings(embeddings)

        return self.mlp(embeddings)

    def _aggregate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate embeddings from multiple layers according to the aggregation
        method.

        Args:
            embeddings: Tensor of shape (batch_size, num_layers, embedding_dim)

        Returns:
            Aggregated embeddings of shape (batch_size, embedding_dim)

        Raises:
            ValueError: If aggregation method is not supported
        """
        if self.aggregation == "mean":
            return embeddings.mean(dim=1)
        elif self.aggregation == "max":
            return embeddings.max(dim=1)[0]
        elif self.aggregation == "concat":
            # Flatten all layer embeddings
            batch_size = embeddings.shape[0]
            return embeddings.view(batch_size, -1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
