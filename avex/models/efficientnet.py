"""EfficientNet model implementation for audio classification.

This module provides EfficientNet model implementations for audio classification
tasks, including B0 and B1 variants with audio-specific preprocessing.
"""

import logging
from typing import List, Optional, Union

import torch
import torch.utils.checkpoint
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, efficientnet_b0, efficientnet_b1

from avex.configs import AudioConfig
from avex.models.base_model import ModelBase

logger = logging.getLogger(__name__)


# EfficientNet (B0/B1). Each class should be called "Model."
class Model(ModelBase):
    """EfficientNet model for audio classification.

    Implements EfficientNet B0/B1 variants for audio classification tasks
    with audio-specific preprocessing and feature extraction.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pretrained: bool = True,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        efficientnet_variant: str = "b0",
    ) -> None:
        # Call parent initializer with audio config
        super().__init__(device=device, audio_config=audio_config)

        # Validate num_classes: required when return_features_only=False
        # (though EfficientNet uses torchvision's classifier which has fixed classes)
        # When return_features_only=True, num_classes is not used
        if not return_features_only and num_classes is None:
            # Use default from torchvision EfficientNet
            num_classes = 1000

        # Store the flag and config
        self.return_features_only = return_features_only
        self.gradient_checkpointing = False
        self.audio_config = audio_config

        # Load the appropriate EfficientNet variant based on configuration
        # Use weights parameter instead of deprecated pretrained parameter
        if efficientnet_variant == "b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = efficientnet_b0(weights=weights)
        elif efficientnet_variant == "b1":
            weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = efficientnet_b1(weights=weights)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {efficientnet_variant}")

        # Move model to device
        self.model = self.model.to(self.device)

        # -------------------------------------------------------------- #
        #  Pre-discover convolutional layers for efficient hook management #
        # -------------------------------------------------------------- #
        # Convolutional layers will be discovered in _discover_linear_layers override

    def _discover_linear_layers(self) -> None:
        """Discover and cache only the EfficientNet layers that are useful
        for embeddings.

        This method is called when target_layers=["all"] is used.
        Specifically filters for:
        - Initial conv layer (model.features.0.0)
        - Final projection layers from each block (model.features.X.Y.block.3.0)
        - Final conv layer (model.features.8.0)
        - Excludes expansion layers, depthwise convs, SE layers, and avgpool
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            for name, _module in self.named_modules():
                # Keep the initial conv layer
                if name == "model.features.0.0":
                    self._layer_names.append(name)

                # Keep final projection layers (last conv in each MBConv block)
                # Pattern: model.features.X.Y.block.3.0
                elif name.endswith(".block.3.0") and "model.features." in name:
                    self._layer_names.append(name)

                # Keep the final conv layer
                elif name == "model.features.8.0":
                    self._layer_names.append(name)

            logger.info(
                f"Discovered {len(self._layer_names)} embedding-relevant layers "
                f"in EfficientNet model: "
                f"{self._layer_names}"
            )

    def _discover_embedding_layers(self) -> None:
        """Discover and cache only the EfficientNet layers that are useful
        for embeddings.

        Specifically filters for:
        - Initial conv layer (model.features.0.0)
        - Final projection layers from each block (model.features.X.Y.block.3.0)
        - Final conv layer (model.features.8.0)
        - Excludes expansion layers, depthwise convs, SE layers, and avgpool
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []

            for name, _module in self.named_modules():
                # Keep the initial conv layer
                if name == "model.features.0.0":
                    self._layer_names.append(name)

                # Keep final projection layers (last conv in each MBConv block)
                # Pattern: model.features.X.Y.block.3.0
                elif name.endswith(".block.3.0") and "model.features." in name:
                    self._layer_names.append(name)

                # Keep the final conv layer
                elif name == "model.features.8.0":
                    self._layer_names.append(name)

            logger.info(
                f"Discovered {len(self._layer_names)} embedding-relevant layers "
                f"in EfficientNet model: "
                f"{self._layer_names}"
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
        return torch.utils.checkpoint.checkpoint(self.model.features, x, use_reentrant=False)

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
            Model output (logits or unpooled features based on init flag)
            - If return_features_only=True: spatial feature maps (B, C, H, W)
            - If return_features_only=False: classification logits (B, num_classes)
        """
        # Process audio
        x = self.process_audio(x)

        # Extract features with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            features = self._checkpointed_features(x)
        else:
            features = self.model.features(x)

        # Return unpooled spatial features if requested
        if self.return_features_only:
            return features

        pooled_features = self.model.avgpool(features)
        flattened_features = torch.flatten(pooled_features, 1)
        logits = self.model.classifier(flattened_features)
        return logits

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "none",
        freeze_backbone: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract embeddings from all registered hooks in the EfficientNet model.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav'
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input (ignored for EfficientNet)
        aggregation : str
            Aggregation method for multiple layers ('mean', 'max', 'cls_token', 'none').
            When 'none', 4D embeddings (B, C, H, W) are reshaped to 3D (B, H, C*W)
            for sequence probe compatibility.
        freeze_backbone : bool
            Whether to freeze the backbone and use torch.no_grad()

        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Model embeddings (tensor if aggregated, list of 3D tensors if
            aggregation='none')

        Raises
        ------
        ValueError
            If no hooks are registered or no outputs are captured
        """
        # Check if hooks are registered
        if not self._hooks:
            raise ValueError("No hooks are registered in the model.")

        # Clear previous hook outputs
        self._clear_hook_outputs()

        try:
            # Process input
            if isinstance(x, dict):
                wav = x["raw_wav"]
            else:
                wav = x

            # Forward pass to trigger hooks (conditionally use torch.no_grad based on
            # freeze_backbone)
            if freeze_backbone:
                with torch.no_grad():
                    self.forward(wav, padding_mask)
            else:
                self.forward(wav, padding_mask)

            # Collect embeddings from hook outputs
            embeddings = []
            for layer_name in self._hook_outputs:
                if layer_name in self._hook_outputs:
                    embeddings.append(self._hook_outputs[layer_name])
                    logger.debug(f"Found embedding for {layer_name}: {self._hook_outputs[layer_name].shape}")
                else:
                    logger.warning(f"No output captured for layer: {layer_name}")

            logger.debug(f"Collected {len(embeddings)} embeddings")

            # Check if we got any embeddings
            if not embeddings:
                raise ValueError("No outputs were captured from registered hooks.")

            logger.debug(f" Aggregation: {aggregation}")
            # Process embeddings based on aggregation parameter
            if aggregation == "none":
                if len(embeddings) == 1:
                    return embeddings[0]
                else:
                    return embeddings
            else:
                logger.debug(f"Using aggregation method: {aggregation}")
                # Average over time and concatenate
                for i in range(len(embeddings)):
                    if embeddings[i].dim() == 2:
                        pass
                    else:
                        if aggregation == "mean":
                            embeddings[i] = embeddings[i].mean(dim=-1)
                        elif aggregation == "max":
                            embeddings[i] = embeddings[i].max(dim=-1)[0]  # max returns (values, indices)
                        elif aggregation == "cls_token":
                            embeddings[i] = embeddings[i][:, 0, :]
                        else:
                            raise ValueError(f"Unsupported aggregation method: {aggregation}")
                        if embeddings[i].dim() == 3:
                            batch_size, channels, width = embeddings[i].shape
                            embeddings[i] = embeddings[i].view(batch_size, -1)
                        else:
                            raise ValueError(
                                f"Unexpected embedding dimension: {embeddings[i].dim()}. Expected 2, 3, or 4."
                            )
                # Concatenate all embeddings
                embeddings = torch.cat(embeddings, dim=1)
                logger.debug(f"Returning concatenated embeddings: {embeddings.shape}")
                return embeddings

        finally:
            # Clear hook outputs for next call
            self._clear_hook_outputs()
