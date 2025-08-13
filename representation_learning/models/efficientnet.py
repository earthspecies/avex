import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torchvision.models import efficientnet_b0, efficientnet_b1

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase

logger = logging.getLogger(__name__)


# EfficientNet (B0/B1). Each class should be called "Model."
class Model(ModelBase):
    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        efficientnet_variant: str = "b0",
    ) -> None:
        # Call parent initializer with audio config
        super().__init__(device=device, audio_config=audio_config)

        # Store the flag and config
        self.return_features_only = return_features_only
        self.gradient_checkpointing = False
        self.audio_config = audio_config

        # Load the appropriate EfficientNet variant based on configuration
        if efficientnet_variant == "b0":
            self.model = efficientnet_b0(pretrained=pretrained)
        elif efficientnet_variant == "b1":
            self.model = efficientnet_b1(pretrained=pretrained)
        else:
            raise ValueError(
                f"Unsupported EfficientNet variant: {efficientnet_variant}"
            )

        # Move model to device
        self.model = self.model.to(self.device)

        # Modify the classifier only if not returning features and num_classes differs.
        if not self.return_features_only and num_classes != 1000:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        # No need to modify classifier if return_features_only is True

        # -------------------------------------------------------------- #
        #  Pre-discover convolutional layers for efficient hook management #
        # -------------------------------------------------------------- #
        self._conv_layer_names: List[str] = []
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self._conv_layer_names.append(name)
        logger.debug(
            f"Discovered {len(self._conv_layer_names)} convolutional layers "
            f"for hook management"
        )

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio input and adapt it for EfficientNet's 3-channel input.

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor

        Returns
        -------
        torch.Tensor
            Processed audio tensor with 3 channels for EfficientNet (B0/B1)
        """
        # First use parent's audio processing
        x = super().process_audio(x)

        # EfficientNet expects 3 channels, so we need to repeat the spectrogram
        if x.dim() == 3:  # (batch, freq, time)
            x = x.unsqueeze(1)  # Add channel dimension
            x = x.repeat(1, 3, 1, 1)  # Repeat across channels

        return x

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for EfficientNet features."""
        self.gradient_checkpointing = True

    def _checkpointed_features(self, x: torch.Tensor) -> torch.Tensor:
        """Apply features with gradient checkpointing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Features extracted with gradient checkpointing
        """
        return torch.utils.checkpoint.checkpoint(
            self.model.features, x, use_reentrant=False
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor
        padding_mask : torch.Tensor
            Padding mask for the input

        Returns
        -------
        torch.Tensor
            Model output (logits or features based on init flag)
        """
        # Process audio
        x = self.process_audio(x)

        # Extract features with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            features = self._checkpointed_features(x)
        else:
            features = self.model.features(x)

        pooled_features = self.model.avgpool(features)
        flattened_features = torch.flatten(pooled_features, 1)

        # Return features or logits based on the flag
        if self.return_features_only:
            return flattened_features
        else:
            logits = self.model.classifier(flattened_features)
            return logits

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        layers: List[str],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        average_over_time: bool = True,
        aggregation: str = "mean",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract embeddings from the model with automatic batch splitting.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav'
        layers : List[str]
            List of layer names. If 'all' is included, all convolutional layers in the
            model will be automatically found and used.
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input (ignored for EfficientNet)
        average_over_time : bool
            Whether to average embeddings over time dimension
        aggregation : str
            Aggregation method for multiple layers ('mean', 'max', 'cls_token', 'none')

        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Model embeddings (tensor if average_over_time=True, list if False)

        Raises
        ------
        ValueError
            If no layers are found matching the specified layer names.
        """
        # Clear previous hook outputs
        self._clear_hook_outputs()

        # Handle 'all' case - use cached convolutional layers
        target_layers = layers.copy()
        if "all" in layers:
            logger.debug(
                "'all' specified in layers, using top 3 convolutional layers "
                "to avoid excessive embedding dimensions..."
            )
            if self._conv_layer_names:
                # Use only the top 3 layers to keep embedding dimensions manageable
                top_layers = (
                    self._conv_layer_names[-3:]
                    if len(self._conv_layer_names) >= 3
                    else self._conv_layer_names
                )
                logger.debug(
                    f"Using top {len(top_layers)} convolutional layers: {top_layers}"
                )
                # Replace 'all' with the top 3 convolutional layer names
                target_layers = [
                    layer for layer in layers if layer != "all"
                ] + top_layers
                logger.debug(
                    f"Target layers after 'all' expansion: {len(target_layers)} layers"
                )
            else:
                logger.warning("No convolutional layers found in the model")

        # Register hooks for requested layers (only if not already registered)
        self._register_hooks_for_layers(target_layers)

        # Store original training state and set to eval for deterministic results
        was_training = self.training
        self.eval()

        # Store original gradient checkpointing state and disable it during extraction
        was_gradient_checkpointing = self.gradient_checkpointing
        self.gradient_checkpointing = False

        # Set deterministic behavior for CUDA if available
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            # Process audio input
            if isinstance(x, dict):
                x = x["raw_wav"]
            x = self.process_audio(x)

            # Extract features with optional gradient checkpointing
            if self.gradient_checkpointing and was_training:
                features = self._checkpointed_features(x)
            else:
                features = self.model.features(x)

            pooled_features = self.model.avgpool(features)
            flattened_features = torch.flatten(pooled_features, 1)

            # If no specific layers requested, return the main features
            if not target_layers:
                return flattened_features

            # Use hook-based approach for specific layers
            logger.debug(f"Starting forward pass with target layers: {target_layers}")

            # The forward pass through features and classifier will trigger hooks
            # for both convolutional layers (in features) and linear layers
            # (in classifier)
            if not self.return_features_only:
                _ = self.model.classifier(flattened_features)
            else:
                # In features_only mode, we still need to trigger the classifier
                # to get hook outputs for any linear layers, but we don't use the result
                _ = self.model.classifier(flattened_features)

            logger.debug(
                f"Forward pass completed. Hook outputs: "
                f"{list(self._hook_outputs.keys())}"
            )

            # Collect embeddings from hook outputs
            embeddings = []
            logger.debug(
                f"Collecting embeddings from {len(target_layers)} target layers"
            )
            for layer_name in target_layers:
                if layer_name in self._hook_outputs:
                    embeddings.append(self._hook_outputs[layer_name])
                    logger.debug(
                        f"Found embedding for {layer_name}: "
                        f"{self._hook_outputs[layer_name].shape}"
                    )
                else:
                    logger.warning(f"No output captured for layer: {layer_name}")

            logger.debug(f"Collected {len(embeddings)} embeddings")

            # Check if we got any embeddings
            if not embeddings:
                raise ValueError(f"No layers found matching: {target_layers}")

            # Process embeddings - flatten all to same dimension and concatenate
            result = []
            for emb in embeddings:
                if emb.dim() == 2:
                    # Already flattened, just append
                    result.append(emb)
                elif emb.dim() == 3:
                    # (B, C, T) -> flatten to (B, C*T)
                    batch_size, channels, time = emb.shape
                    flattened = emb.view(batch_size, -1)
                    result.append(flattened)
                elif emb.dim() == 4:
                    # (B, C, H, W) -> flatten to (B, C*H*W)
                    batch_size, channels, height, width = emb.shape
                    flattened = emb.view(batch_size, -1)
                    result.append(flattened)
                else:
                    raise ValueError(
                        f"Unexpected embedding dimension: {emb.dim()}. "
                        f"Expected 2, 3, or 4."
                    )

            if average_over_time:
                # Apply aggregation across layers
                if len(result) > 1:
                    if aggregation == "mean":
                        # Concatenate all flattened embeddings along the last dimension
                        return torch.cat(result, dim=1)
                    elif aggregation == "max":
                        # Concatenate all flattened embeddings along the last dimension
                        return torch.cat(result, dim=1)
                    elif aggregation == "cls_token":
                        # Concatenate all flattened embeddings along the last dimension
                        return torch.cat(result, dim=1)
                    elif aggregation == "none":
                        # Try to stack, but if sizes don't match, fall back to
                        # concatenation
                        try:
                            return torch.stack(result, dim=1)
                        except RuntimeError:
                            # If stacking fails due to different sizes, concatenate
                            # instead
                            return torch.cat(result, dim=1)
                    else:
                        raise ValueError(f"Unknown aggregation method: {aggregation}")
                else:
                    return result[0]
            else:
                # Return list of embeddings without concatenation
                if len(result) > 1 and aggregation == "none":
                    # Stack along a new dimension to preserve layer information
                    return torch.stack(result, dim=1)
                else:
                    return result

        finally:
            # Restore original training state
            if was_training:
                self.train()
            # Restore gradient checkpointing state
            self.gradient_checkpointing = was_gradient_checkpointing
            # Restore CUDA settings
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
            # Clear hook outputs for next call
            self._clear_hook_outputs()
