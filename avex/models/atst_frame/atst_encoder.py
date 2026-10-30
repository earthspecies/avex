"""ATST Frame encoder implementation.

This module provides the ATST (Audio Spectrogram Transformer) Frame encoder
implementation for audio representation learning tasks.
"""

from typing import Optional

import torch
import torch.nn as nn

from avex.configs import AudioConfig
from avex.models.base_model import ModelBase

from .atst_frame import get_timestamp_embedding, load_model


class Model(ModelBase):
    """ATST Frame model following the repository's standard model interface."""

    def __init__(
        self,
        atst_model_path: str,
        num_classes: Optional[int] = None,
        pretrained: bool = True,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = True,
    ) -> None:
        """Initialize ATST model.

        Parameters
        ----------
        atst_model_path : str
            Path to the pretrained ATST model
        num_classes : Optional[int], optional
            Number of output classes (None or ignored when return_features_only=True)
        pretrained : bool, optional
            Whether to use pretrained weights (always True for ATST)
        device : str, optional
            Device to run the model on
        audio_config : Optional[AudioConfig], optional
            Audio preprocessing configuration
        return_features_only : bool, optional
            Whether to return features only (no classification head)
        """
        # Call parent initializer with audio config
        super().__init__(device=device, audio_config=audio_config)

        # Validate num_classes: required when return_features_only=False
        if not return_features_only and num_classes is None:
            # Use default (1000 classes)
            num_classes = 1000

        # Store configuration
        self.atst_model_path = atst_model_path
        self.return_features_only = return_features_only
        self.num_classes = num_classes

        # Load the ATST model
        target_device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.atst = load_model(atst_model_path, device=target_device)

        # Add classification head if needed
        if not return_features_only:
            # Get the feature dimension from ATST output
            # We'll determine this dynamically or use a reasonable default
            self.classifier = nn.Linear(4608, num_classes)  # 768 is common ATST dimension

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through ATST encoder.

        Args:
            x : torch.Tensor
                Input audio tensor
            padding_mask : torch.Tensor
                Padding mask for the input (unused by ATST but kept for interface
                compliance)

        Returns
            torch.Tensor: Audio representation tensor
        """
        # Process audio using the base class method
        x = self.process_audio(x)

        # Get timestamp embeddings from ATST
        encoding = get_timestamp_embedding(x, self.atst)

        # Return features or apply classifier
        if self.return_features_only:
            # Average over time dimension if present
            if encoding.dim() == 3:  # (batch, features, time)
                encoding = torch.mean(encoding, dim=-1)  # Average over time
            return encoding
        else:
            # Average over time dimension and apply classifier
            if encoding.dim() == 3:
                encoding = torch.mean(encoding, dim=-1)
            logits = self.classifier(encoding)
            return logits

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "none",
        freeze_backbone: bool = True,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Extract embeddings using the forward pass.

        ATST yields a single time-averaged (B, D) embedding, so every
        aggregation method returns that tensor and ``"none"`` wraps it as a
        one-element list of shape (B, 1, D) for sequence probes.

        Args:
            x: Input tensor or dictionary containing 'raw_wav' and 'padding_mask'
            padding_mask: Optional padding mask tensor
            aggregation: One of "none", "mean", "max", "cls_token"
            freeze_backbone: When True, run the backbone under ``torch.no_grad()``

        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            ATST embeddings from the model.
        """
        # Handle input format
        if isinstance(x, dict):
            raw_wav = x["raw_wav"]
            p_mask = x.get("padding_mask", padding_mask)
        else:
            raw_wav = x
            p_mask = padding_mask

        if p_mask is None:
            p_mask = torch.zeros(
                raw_wav.size(0),
                raw_wav.size(1),
                device=raw_wav.device,
                dtype=torch.bool,
            )

        if freeze_backbone:
            with torch.no_grad():
                embeddings = self.forward(raw_wav, p_mask)
        else:
            embeddings = self.forward(raw_wav, p_mask)
        return self._aggregate_single_embedding(embeddings, aggregation)

    def freeze(self) -> None:
        """Freeze the ATST backbone parameters."""
        if hasattr(self.atst, "freeze"):
            self.atst.freeze()
        else:
            # Fallback: manually freeze parameters
            for param in self.atst.parameters():
                param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze the ATST backbone parameters."""
        if hasattr(self.atst, "unfreeze"):
            self.atst.unfreeze()
        else:
            # Fallback: manually unfreeze parameters
            for param in self.atst.parameters():
                param.requires_grad = True
