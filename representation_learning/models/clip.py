import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from representation_learning.models.base_model import ModelBase
from representation_learning.models.efficientnet import (
    Model as EfficientNet,
)


class CLIPModel(ModelBase):
    """CLAP model combining EfficientNet for audio and RoBERTa for text."""

    def __init__(
        self,
        device: str,
        audio_config: Optional[Dict[str, Any]] = None,
        text_model_name: str = "roberta-base",
        projection_dim: int = 512,
        temperature: float = 0.07,
        efficientnet_variant: str = "b0",
    ) -> None:
        super().__init__(device, audio_config)

        self.audio_encoder = EfficientNet(
            device=device,
            audio_config=audio_config,
            return_features_only=True,  # Get features before classifier
            efficientnet_variant=efficientnet_variant,
        )

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Projection heads: two-layer MLP (Linear → ReLU → Linear)
        # EfficientNet B0 has 1280 features, B1 has 1280 features too
        audio_feature_dim = 1280  # Both B0 and B1 have the same feature dimension
        hidden_dim = projection_dim
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
        text_feature_dim = self.text_encoder.config.hidden_size
        self.text_projection = nn.Sequential(
            nn.Linear(text_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        init_value = torch.log(torch.tensor(1.0 / temperature))
        self.logit_scale = torch.nn.Parameter(init_value)

        # Move models to device
        self.audio_encoder.to(device)
        self.text_encoder.to(device)
        self.audio_projection.to(device)
        self.text_projection.to(device)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for both audio and text encoders."""
        # Enable checkpointing for audio encoder (EfficientNet)
        self.audio_encoder.enable_gradient_checkpointing()

        # Enable checkpointing for text encoder (HuggingFace transformer)
        self.text_encoder.gradient_checkpointing_enable()

    def encode_audio(
        self, audio: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode audio input using EfficientNet.

        Parameters
        ----------
        audio : torch.Tensor
            Audio input tensor

        Returns
        -------
        torch.Tensor
            Normalized audio embeddings
        """
        features = self.audio_encoder(audio, padding_mask)
        return F.normalize(self.audio_projection(features), dim=-1)

    def encode_text(self, text: list[str]) -> torch.Tensor:
        """Encode text input using RoBERTa.

        Parameters
        ----------
        text : list[str]
            List of text strings to encode

        Returns
        -------
        torch.Tensor
            Normalized text embeddings
        """
        current_device = next(self.parameters()).device
        tokens = self.text_tokenizer(
            text, padding=True, truncation=True, max_length=70, return_tensors="pt"
        ).to(current_device)

        outputs = self.text_encoder(**tokens)
        features = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        return F.normalize(self.text_projection(features), dim=-1)

    def forward(
        self, audio: torch.Tensor, text: list[str], padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing audio and text embeddings.

        Args:
            audio: Audio tensor of shape (batch_size, time_steps)
            text: List of text strings of length batch_size

        Returns:
            Tuple of (audio_embeddings, text_embeddings, logit_scale)
        """
        # Get normalized embeddings
        audio_embeddings = self.encode_audio(audio, padding_mask)
        text_embeddings = self.encode_text(text)

        LOGIT_SCALE_MAX = math.log(1.0 / 0.01)  # log(100)
        with torch.no_grad():
            self.logit_scale.clamp_(max=LOGIT_SCALE_MAX)

        return audio_embeddings, text_embeddings, self.logit_scale.exp()

    def extract_embeddings(
        self,
        x: torch.Tensor | Dict[str, torch.Tensor],
        layers: list[str],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        average_over_time: bool = True,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """Extract audio embeddings from the CLIP model.

        Parameters
        ----------
        x : torch.Tensor | Dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav' and 'padding_mask'
        layers : list[str]
            List of layer names (kept for interface compatibility but ignored)
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input
        average_over_time : bool
            Kept for interface compatibility but ignored
        aggregation : str
            Kept for interface compatibility but ignored

        Returns
        -------
        torch.Tensor
            Projected audio embeddings suitable for contrastive learning
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

        # Extract audio features and apply projection
        audio_features = self.audio_encoder(raw_wav, p_mask)
        projected_features = self.audio_projection(audio_features)

        return projected_features
