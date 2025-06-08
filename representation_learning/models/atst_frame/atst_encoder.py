from typing import List, Optional

import torch
import torch.nn as nn

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase

from .atst_frame import get_timestamp_embedding, load_model


class Model(ModelBase):
    """ATST Frame model following the repository's standard model interface."""

    def __init__(
        self,
        atst_model_path: str,
        num_classes: int = 1000,
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
        num_classes : int, optional
            Number of output classes (ignored when return_features_only=True)
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

        # Store configuration
        self.atst_model_path = atst_model_path
        self.return_features_only = return_features_only
        self.num_classes = num_classes

        # Load the ATST model
        target_device = (
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.atst = load_model(atst_model_path, device=target_device)

        # Add classification head if needed
        if not return_features_only:
            # Get the feature dimension from ATST output
            # We'll determine this dynamically or use a reasonable default
            self.classifier = nn.Linear(
                4608, num_classes
            )  # 768 is common ATST dimension

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ATST model.

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor
        padding_mask : torch.Tensor
            Padding mask for the input (unused by ATST but kept for interface compliance)

        Returns
        -------
        torch.Tensor
            Model output (features or logits based on return_features_only flag)
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
        layers: List[str],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        average_over_time: bool = True,
    ) -> torch.Tensor:
        """
        Extract embeddings using the forward pass.

        For ATST, we ignore the layers parameter and always return the final embeddings.

        Args:
            x: Input tensor or dictionary containing 'raw_wav' and 'padding_mask'
            layers: List of layer names (ignored for ATST)
            padding_mask: Optional padding mask tensor
            average_over_time: Whether to average over time dimension

        Returns
        -------
        torch.Tensor
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

        # Get embeddings using forward pass
        embeddings = self.forward(raw_wav, p_mask)
        return embeddings

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
