import logging
from typing import List, Optional

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

        # Store the flag
        self.return_features_only = return_features_only
        self.gradient_checkpointing = False

        # Load the appropriate EfficientNet variant based on configuration
        if efficientnet_variant == "b0":
            self.model = efficientnet_b0(pretrained=pretrained)
        elif efficientnet_variant == "b1":
            self.model = efficientnet_b1(pretrained=pretrained)
        else:
            raise ValueError(
                f"Unsupported EfficientNet variant: {efficientnet_variant}"
            )

        # Modify the classifier only if not returning features and num_classes differs.
        if not self.return_features_only and num_classes != 1000:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        # No need to modify classifier if return_features_only is True

        # -------------------------------------------------------------- #
        #  Pre-discover linear layers for efficient hook management      #
        # -------------------------------------------------------------- #
        self._linear_layer_names: List[str] = []
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                self._linear_layer_names.append(name)
        logger.info(
            f"Discovered {len(self._linear_layer_names)} linear layers "
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
    ) -> torch.Tensor:
        """Extract embeddings from the model.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav'
        layers : List[str]
            List of layer names. If 'all' is included, all linear layers in the model
            will be automatically found and used.
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input (ignored for EfficientNet)
        average_over_time : bool
            Whether to average embeddings over time dimension

        Returns
        -------
        torch.Tensor
            Model embeddings

        Raises
        ------
        ValueError
            If no layers are found matching the specified layer names.
        """
        # Clear previous hook outputs
        self._clear_hook_outputs()

        # Handle 'all' case - use cached linear layers
        target_layers = layers.copy()
        if "all" in layers:
            if not self.return_features_only:
                logger.info(
                    "'all' specified in layers, using pre-discovered linear layers..."
                )
                if self._linear_layer_names:
                    logger.info(
                        f"Using {len(self._linear_layer_names)} pre-discovered "
                        f"linear layers"
                    )
                    # Replace 'all' with the actual linear layer names
                    target_layers = [
                        layer for layer in layers if layer != "all"
                    ] + self._linear_layer_names
                    logger.info(
                        f"Target layers after 'all' expansion: "
                        f"{len(target_layers)} layers"
                    )
                else:
                    logger.warning("No linear layers found in the model")
            else:
                # In features_only mode, 'all' just means the main features
                target_layers = [layer for layer in layers if layer != "all"]

        # Register hooks for requested layers (only if not already registered)
        self._register_hooks_for_layers(target_layers)

        try:
            # Process audio input
            if isinstance(x, dict):
                x = x["raw_wav"]
            x = self.process_audio(x)

            # Extract features with optional gradient checkpointing
            if self.gradient_checkpointing and self.training:
                features = self._checkpointed_features(x)
            else:
                features = self.model.features(x)

            pooled_features = self.model.avgpool(features)
            flattened_features = torch.flatten(pooled_features, 1)

            # If no specific layers requested or only 'all' is requested,
            # return the main features
            if not target_layers or (
                len(target_layers) == 1 and target_layers[0] == "all"
            ):
                return flattened_features

            # Otherwise, use hook-based approach for specific layers
            logger.debug(f"Starting forward pass with target layers: {target_layers}")

            # Apply classifier if not returning features only (to trigger hooks)
            if not self.return_features_only:
                _ = self.model.classifier(flattened_features)
            else:
                # In features_only mode, we need to manually trigger the classifier
                # to get hook outputs
                # but we don't use the result
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

            if average_over_time:
                result = []
                for emb in embeddings:
                    if emb.dim() == 2:
                        # Already in correct shape, just append
                        result.append(emb)
                    elif emb.dim() == 3:
                        aggregated = torch.mean(emb, dim=1)
                        result.append(aggregated)
                    else:
                        raise ValueError(
                            f"Unexpected embedding dimension: {emb.dim()}. "
                            f"Expected 2 or 3."
                        )
                return torch.cat(result, dim=1)
            else:
                return embeddings

        finally:
            # Clear hook outputs for next call
            self._clear_hook_outputs()
