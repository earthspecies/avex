import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from representation_learning.models.base_model import ModelBase
from representation_learning.models.efficientnetb0 import (
    Model as EfficientNetB0,
)


class CLIPModel(ModelBase):
    """CLIP-like model combining EfficientNetB0 for audio and RoBERTa for text."""

    def __init__(
        self,
        device: str,
        audio_config: Optional[Dict[str, Any]] = None,
        text_model_name: str = "roberta-base",
        projection_dim: int = 512,
        temperature: float = 0.07,
    ) -> None:
        super().__init__(device, audio_config)

        # Initialize audio encoder (EfficientNetB0)
        self.audio_encoder = EfficientNetB0(
            device=device,
            audio_config=audio_config,
            return_features_only=True,  # Get features before classifier
        )

        # Initialize text encoder (RoBERTa)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Projection heads: two-layer MLP (Linear → ReLU → Linear)
        audio_feature_dim = 1280
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

        # Learnable log-logit scale parameter as in original CLIP implementation
        # Start from log(1/temperature) so that exp(logit_scale) == 1/temperature.
        init_value = torch.log(torch.tensor(1.0 / temperature))
        self.logit_scale = torch.nn.Parameter(init_value)

        # Move models to device
        self.audio_encoder.to(device)
        self.text_encoder.to(device)
        self.audio_projection.to(device)
        self.text_projection.to(device)

    def encode_audio(
        self, audio: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode audio input using EfficientNetB0.

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
        # Move token tensors to *current* device of the module (safe for DDP)
        current_device = next(self.parameters()).device
        tokens = self.text_tokenizer(
            text, padding=True, truncation=True, max_length=50, return_tensors="pt"
        ).to(current_device)

        # Get text embeddings
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

        # Clamp temperature as in the original CLIP paper (<= log(100) ≈ 4.605).
        LOGIT_SCALE_MAX = math.log(1.0 / 0.01)  # log(100)
        with torch.no_grad():
            self.logit_scale.clamp_(max=LOGIT_SCALE_MAX)

        # Return embeddings and *scalar* positive logit scale so the loss can
        return audio_embeddings, text_embeddings, self.logit_scale.exp()
