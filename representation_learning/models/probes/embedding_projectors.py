"""Embedding projectors for handling multiple embedding shapes in probes."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Conv4DProjector(nn.Module):
    """
    Projects 4D convolutional embeddings to 3D sequence format.

    Converts 4D embeddings from convolutional layers (like EfficientNet) to 3D
    format suitable for sequence-based probes. Uses width dimension as sequence
    length and projects channels and height into feature dimension.

    Input: (batch, channels, height, width)
    Output: (batch, width, target_feature_dim) or (batch, width, channels*height)

    Args:
        target_feature_dim: Optional target feature dimension. If None, uses
            channels*height from input. If specified, projects to the target dimension.

        use_parameter_free: If True, uses parameter-free projection with pooling
            and interpolation instead of learnable weights.
    """

    def __init__(
        self,
        target_feature_dim: Optional[int] = None,
        target_sequence_length: Optional[int] = None,
        use_parameter_free: bool = False,
    ) -> None:
        super().__init__()
        self.target_feature_dim = target_feature_dim
        self.target_sequence_length = target_sequence_length
        self.use_parameter_free = use_parameter_free
        self.conv1x1: Optional[nn.Conv2d] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project 4D tensor to 3D sequence format.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Projected tensor of shape (batch, width, target_feature_dim) or
            (batch, width, channels*height) if no target dimension specified

        Raises:
            ValueError: If input tensor is not 4D
        """
        if x.dim() != 4:
            raise ValueError(
                f"Conv4DProjector expects 4D input (batch, channels, height, width), "
                f"got shape {x.shape} with {x.dim()} dimensions"
            )

        batch_size, channels, height, width = x.shape

        # Use parameter-free projection if requested
        if self.use_parameter_free:
            return self._project_4d_to_sequence_param_free(x, self.target_sequence_length, self.target_feature_dim)

        # Apply 1x1 convolution if target_feature_dim is specified
        if self.target_feature_dim is not None:
            # Check if we need to do any projection
            current_feature_dim = channels * height
            current_seq_len = width

            # If feature dimensions already match target, just do format conversion
            if current_feature_dim == self.target_feature_dim:
                # No feature projection needed - just reshape to 3D format
                x_output = x.transpose(1, 3).reshape(batch_size, width, channels * height)

                # Handle target sequence length if specified
                if self.target_sequence_length is not None and current_seq_len != self.target_sequence_length:
                    # Use interpolation to resize sequence length
                    # x_output shape: (batch, seq_len, features)
                    # We need to interpolate along the sequence dimension (dim=1)
                    x_output = torch.nn.functional.interpolate(
                        x_output.transpose(1, 2),  # (batch, features, seq_len)
                        size=self.target_sequence_length,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)  # Back to (batch, seq_len, features)
            else:
                # Create 1x1 conv layer if needed
                if (
                    self.conv1x1 is None
                    or self.conv1x1.in_channels != channels * height
                    or self.conv1x1.out_channels != self.target_feature_dim
                ):
                    self.conv1x1 = nn.Conv2d(
                        in_channels=channels * height,
                        out_channels=self.target_feature_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    ).to(x.device)

                # Reshape input for 1x1 conv: (batch, channels, height, width) ->
                # (batch, channels*height, 1, width)
                x_reshaped = x.reshape(batch_size, channels * height, 1, width)

                # Apply 1x1 convolution: (batch, channels*height, 1, width) ->
                # (batch, target_feature_dim, 1, width)
                x_conv = self.conv1x1(x_reshaped)

                # Reshape to get width as sequence:
                # (batch, target_feature_dim, 1, width)
                # -> (batch, width, target_feature_dim)
                x_output = x_conv.squeeze(2).transpose(1, 2)  # Remove height dim and transpose

                # Handle target sequence length if specified
                if self.target_sequence_length is not None:
                    current_seq_len = x_output.shape[1]
                    if current_seq_len != self.target_sequence_length:
                        # Use interpolation to resize sequence length
                        # x_output shape: (batch, seq_len, features)
                        # We need to interpolate along the sequence dimension (dim=1)
                        x_output = torch.nn.functional.interpolate(
                            x_output.transpose(1, 2),  # (batch, features, seq_len)
                            size=self.target_sequence_length,
                            mode="linear",
                            align_corners=False,
                        ).transpose(1, 2)  # Back to (batch, seq_len, features)

        else:
            # No target dimension - use simple reshape approach
            # Transpose to (batch, width, channels, height) then reshape
            x_transposed = x.transpose(1, 3)  # (batch, width, height, channels)
            x_output = x_transposed.reshape(batch_size, width, channels * height)

            # Handle target sequence length if specified
            if self.target_sequence_length is not None:
                current_seq_len = x_output.shape[1]
                if current_seq_len != self.target_sequence_length:
                    # Use interpolation to resize sequence length
                    # x_output shape: (batch, seq_len, features)
                    # We need to interpolate along the sequence dimension (dim=1)
                    x_output = torch.nn.functional.interpolate(
                        x_output.transpose(1, 2),  # (batch, features, seq_len)
                        size=self.target_sequence_length,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)  # Back to (batch, seq_len, features)

        logger.debug(f"Conv4DProjector: {x.shape} -> {x_output.shape}")

        return x_output

    def _project_4d_to_sequence_param_free(
        self,
        x: torch.Tensor,
        target_seq_len: Optional[int],
        target_feature_dim: Optional[int],
    ) -> torch.Tensor:
        """
        Parameter-free projection from (B, C, H, W) → (B, T, F)
        using pooling + interpolation (no learnable weights).

        Args:
            x: Input tensor (B, C, H, W)
            target_seq_len: Target sequence length (T)
            target_feature_dim: Target feature dimension (F)

        Returns:
            Tensor of shape (B, T, F)
        """
        B, C, H, W = x.shape

        # 1) Pool spatial H dimension to 1 → reduces (B, C, H, W) → (B, C, 1, W)
        x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, W))  # keeps temporal dimension W

        # 2) Collapse C x 1 to "features": (B, C, 1, W) → (B, C, W)
        x_pooled = x_pooled.squeeze(2)  # (B, C, W)

        # 3) Interpolate along the channel dimension C → F
        if target_feature_dim is not None and x_pooled.shape[1] != target_feature_dim:
            x_features = torch.nn.functional.interpolate(
                x_pooled.transpose(1, 2),  # (B, W, C)
                size=target_feature_dim,  # new feature dim
                mode="linear",
                align_corners=False,
            )  # (B, W, F)
        else:
            x_features = x_pooled.transpose(1, 2)  # (B, W, C)

        # 4) Interpolate along temporal dim W → T
        if target_seq_len is not None and x_features.shape[1] != target_seq_len:
            x_features = x_features.transpose(1, 2)  # (B, F, W)
            x_features = torch.nn.functional.interpolate(
                x_features,
                size=target_seq_len,  # new sequence length
                mode="linear",
                align_corners=False,
            )
            x_features = x_features.transpose(1, 2)  # (B, T, F)

        return x_features  # (B, T, F)


class Sequence3DProjector(nn.Module):
    """
    Standardizes 3D embeddings to consistent format.

    Handles different 3D embedding formats and converts them to the standard
    (batch, sequence_length, features) format expected by sequence probes.

    Supported input formats:
    - (batch, sequence_length, features) - already correct
    - (sequence_length, batch, features) - transpose to batch-first
    - (batch, features, sequence_length) - transpose to sequence-last

    Args:
        target_feature_dim: Optional target feature dimension for projection
        use_parameter_free: If True, uses parameter-free projection with interpolation
            instead of learnable weights
    """

    def __init__(
        self,
        target_feature_dim: Optional[int] = None,
        target_sequence_length: Optional[int] = None,
        use_parameter_free: bool = False,
    ) -> None:
        super().__init__()
        self.target_feature_dim = target_feature_dim
        self.target_sequence_length = target_sequence_length
        self.use_parameter_free = use_parameter_free
        self.projection_layer: Optional[nn.Linear] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standardize 3D tensor to (batch, sequence_length, features) format.

        Args:
            x: Input tensor of shape (batch, seq_len, features) or variants

        Returns:
            Standardized tensor of shape (batch, sequence_length, features)

        Raises:
            ValueError: If input tensor is not 3D
        """
        if x.dim() != 3:
            raise ValueError(f"Sequence3DProjector expects 3D input, got shape {x.shape} with {x.dim()} dimensions")

        batch_size, seq_len, features = x.shape
        original_shape = x.shape

        # Use parameter-free projection if requested
        if self.use_parameter_free:
            return self._project_3d_to_sequence_param_free(x, self.target_sequence_length, self.target_feature_dim)

        # Apply projection if needed
        if self.target_feature_dim is not None:
            # Check if we need to do any projection
            if features == self.target_feature_dim:
                # No feature projection needed - dimensions already match
                x_standardized = x

                # Handle target sequence length if specified
                if self.target_sequence_length is not None and seq_len != self.target_sequence_length:
                    # Use interpolation to resize sequence length
                    # x_standardized shape: (batch, seq_len, features)
                    # We need to interpolate along the sequence dimension (dim=1)
                    x_standardized = torch.nn.functional.interpolate(
                        x_standardized.transpose(1, 2),  # (batch, features, seq_len)
                        size=self.target_sequence_length,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)  # Back to (batch, seq_len, features)
            else:
                if self.projection_layer is None or self.projection_layer.in_features != features:
                    self.projection_layer = nn.Linear(features, self.target_feature_dim).to(x.device)
                x_standardized = self.projection_layer(x)
        else:
            x_standardized = x

        # Handle target sequence length if specified
        if self.target_sequence_length is not None:
            current_seq_len = x_standardized.shape[1]
            if current_seq_len != self.target_sequence_length:
                # Use interpolation to resize sequence length
                # x_standardized shape: (batch, seq_len, features)
                # We need to interpolate along the sequence dimension (dim=1)
                x_standardized = torch.nn.functional.interpolate(
                    x_standardized.transpose(1, 2),  # (batch, features, seq_len)
                    size=self.target_sequence_length,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)  # Back to (batch, seq_len, features)

        logger.debug(f"Sequence3DProjector: {original_shape} -> {x_standardized.shape}")

        return x_standardized

    def _project_3d_to_sequence_param_free(
        self,
        x: torch.Tensor,
        target_seq_len: Optional[int],
        target_feature_dim: Optional[int],
    ) -> torch.Tensor:
        """
        Parameter-free projection for 3D tensors using interpolation.

        Args:
            x: Input tensor (B, T, F)
            target_seq_len: Target sequence length (T)
            target_feature_dim: Target feature dimension (F)

        Returns:
            Tensor of shape (B, T, F) with interpolated dimensions
        """
        B, T, F = x.shape

        # Interpolate along sequence dimension first if needed
        if target_seq_len is not None and T != target_seq_len:
            x_seq = torch.nn.functional.interpolate(
                x.transpose(1, 2),  # (B, F, T)
                size=target_seq_len,  # new sequence length
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)  # (B, T, F)
        else:
            x_seq = x

        # Interpolate along feature dimension if needed
        if target_feature_dim is not None and x_seq.shape[2] != target_feature_dim:
            # For feature interpolation, we use a simpler approach:
            # Reshape to (B*T, F), interpolate to (B*T, target_feature_dim), then
            # reshape back
            B, T, F = x_seq.shape
            x_reshaped = x_seq.reshape(B * T, F)  # (B*T, F)

            # Add dummy dimension for interpolation: (B*T, F) -> (B*T, 1, F)
            x_unsqueezed = x_reshaped.unsqueeze(1)  # (B*T, 1, F)

            # Interpolate along the last dimension
            x_interp = torch.nn.functional.interpolate(
                x_unsqueezed,  # (B*T, 1, F)
                size=target_feature_dim,  # new feature dim
                mode="linear",
                align_corners=False,
            )  # (B*T, 1, target_feature_dim)

            # Remove dummy dimension and reshape back
            x_squeezed = x_interp.squeeze(1)  # (B*T, target_feature_dim)
            x_features = x_squeezed.reshape(B, T, target_feature_dim)  # (B, T, target_feature_dim)
        else:
            x_features = x_seq

        return x_features


class EmbeddingProjector(nn.Module):
    """
    Unified projector that automatically detects embedding shape and applies
    appropriate projection.

    This is the main interface for embedding projection in probes. It automatically
    detects the input tensor dimensions and applies the appropriate projection
    strategy.

    Supported input formats:
    - 4D: (batch, channels, height, width) -> (batch, width, channels*height)
    - 3D: Various 3D formats -> (batch, sequence_length, features)
    - 2D: (batch, features) -> (batch, 1, features) for sequence probes

    Args:
        target_feature_dim: Optional target feature dimension for all projections

        force_sequence_format: Whether to force 2D inputs to 3D sequence format
        use_parameter_free: If True, uses parameter-free projection with interpolation
            instead of learnable weights
    """

    def __init__(
        self,
        target_feature_dim: Optional[int] = None,
        target_sequence_length: Optional[int] = None,
        force_sequence_format: bool = True,
        use_parameter_free: bool = False,
    ) -> None:
        super().__init__()
        self.target_feature_dim = target_feature_dim
        self.target_sequence_length = target_sequence_length

        self.force_sequence_format = force_sequence_format
        self.use_parameter_free = use_parameter_free

        # Initialize projectors
        self.conv4d_projector = Conv4DProjector(
            target_feature_dim=target_feature_dim,
            target_sequence_length=target_sequence_length,
            use_parameter_free=use_parameter_free,
        )
        self.seq3d_projector = Sequence3DProjector(
            target_feature_dim=target_feature_dim,
            target_sequence_length=target_sequence_length,
            use_parameter_free=use_parameter_free,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input tensor to appropriate format based on its dimensions.

        Args:
            x: Input tensor of any supported shape

        Returns:
            Projected tensor in appropriate format for sequence probes

        Raises:
            ValueError: If input tensor has unsupported dimensions
        """
        if x.dim() == 4:
            # 4D tensor - use convolutional projector
            return self.conv4d_projector(x)
        elif x.dim() == 3:
            # 3D tensor - use sequence projector
            return self.seq3d_projector(x)
        elif x.dim() == 2:
            # 2D tensor - convert to sequence format if requested
            if self.force_sequence_format:
                batch_size, features = x.shape
                # Add sequence dimension of length 1
                x_3d = x.unsqueeze(1)  # (batch, 1, features)

                # Apply projection if needed
                if self.use_parameter_free:
                    # Use parameter-free projection for 2D case
                    return self._project_2d_to_sequence_param_free(
                        x, self.target_sequence_length, self.target_feature_dim
                    )
                elif self.target_feature_dim is not None:
                    if not hasattr(self, "linear2d_projector") or self.linear2d_projector.in_features != features:
                        self.linear2d_projector = nn.Linear(features, self.target_feature_dim).to(x.device)
                    x_3d = self.linear2d_projector(x_3d)

                # Handle target sequence length if specified
                if self.target_sequence_length is not None:
                    current_seq_len = x_3d.shape[1]
                    if current_seq_len != self.target_sequence_length:
                        # Use interpolation to resize sequence length
                        # x_3d shape: (batch, seq_len, features)
                        # We need to interpolate along the sequence dimension (dim=1)
                        x_3d = torch.nn.functional.interpolate(
                            x_3d.transpose(1, 2),  # (batch, features, seq_len)
                            size=self.target_sequence_length,
                            mode="linear",
                            align_corners=False,
                        ).transpose(1, 2)  # Back to (batch, seq_len, features)

                logger.debug(f"EmbeddingProjector 2D->3D: {x.shape} -> {x_3d.shape}")
                return x_3d
            else:
                # For 2D output, apply projection if needed
                if self.target_feature_dim is not None:
                    batch_size, features = x.shape
                    if not hasattr(self, "linear2d_projector") or self.linear2d_projector.in_features != features:
                        self.linear2d_projector = nn.Linear(features, self.target_feature_dim).to(x.device)
                    x = self.linear2d_projector(x)
                    logger.debug(f"EmbeddingProjector 2D->2D: {x.shape} -> {x.shape}")
                return x
        else:
            raise ValueError(
                f"EmbeddingProjector supports 2D, 3D, and 4D tensors, got shape {x.shape} with {x.dim()} dimensions"
            )

    def get_output_shape_info(self, input_shape: Tuple[int, ...]) -> dict:
        """
        Get information about the output shape for a given input shape.

        Args:
            input_shape: Input tensor shape

        Returns:
            Dictionary with output shape information

        Raises:
            ValueError: If input_shape has unsupported dimensions
        """
        if len(input_shape) == 4:
            batch, channels, height, width = input_shape
            if self.target_feature_dim is not None:
                output_features = self.target_feature_dim
            else:
                output_features = channels * height
            return {
                "input_shape": input_shape,
                "output_shape": (batch, width, output_features),
                "projector_type": "Conv4DProjector",
                "sequence_length": width,
                "feature_dim": output_features,
            }
        elif len(input_shape) == 3:
            batch, dim1, dim2 = input_shape
            if self.target_feature_dim is not None:
                output_features = self.target_feature_dim
            else:
                output_features = max(dim1, dim2)  # Assume larger is features
            return {
                "input_shape": input_shape,
                "output_shape": (batch, min(dim1, dim2), output_features),
                "projector_type": "Sequence3DProjector",
                "sequence_length": min(dim1, dim2),
                "feature_dim": output_features,
            }
        elif len(input_shape) == 2:
            batch, features = input_shape
            if self.force_sequence_format:
                if self.target_feature_dim is not None:
                    output_features = self.target_feature_dim
                else:
                    output_features = features
                return {
                    "input_shape": input_shape,
                    "output_shape": (batch, 1, output_features),
                    "projector_type": "2D->3D",
                    "sequence_length": 1,
                    "feature_dim": output_features,
                }
            else:
                return {
                    "input_shape": input_shape,
                    "output_shape": input_shape,
                    "projector_type": "No projection",
                    "sequence_length": None,
                    "feature_dim": features,
                }
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

    def _project_2d_to_sequence_param_free(
        self,
        x: torch.Tensor,
        target_seq_len: Optional[int],
        target_feature_dim: Optional[int],
    ) -> torch.Tensor:
        """
        Parameter-free projection for 2D tensors to 3D sequence format.

        Args:
            x: Input tensor (B, F)
            target_seq_len: Target sequence length (T)
            target_feature_dim: Target feature dimension (F)

        Returns:
            Tensor of shape (B, T, F) with interpolated dimensions
        """
        B, F = x.shape

        # Convert to 3D with sequence length 1: (B, F) -> (B, 1, F)
        x_3d = x.unsqueeze(1)  # (B, 1, F)

        # Use the 3D parameter-free projection method
        return self.seq3d_projector._project_3d_to_sequence_param_free(x_3d, target_seq_len, target_feature_dim)
