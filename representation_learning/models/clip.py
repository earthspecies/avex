from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from representation_learning.models.base_model import ModelBase
from representation_learning.models.efficientnetb0 import EfficientNetB0


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
        self.audio_encoder = EfficientNetB0(device, audio_config)

        # Initialize text encoder (RoBERTa)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Projection layers
        self.audio_projection = nn.Linear(
            self.audio_encoder.model.classifier.in_features, projection_dim
        )
        self.text_projection = nn.Linear(
            self.text_encoder.config.hidden_size, projection_dim
        )

        # Temperature parameter for contrastive loss
        self.temperature = temperature

        # Move models to device
        self.audio_encoder.to(device)
        self.text_encoder.to(device)
        self.audio_projection.to(device)
        self.text_projection.to(device)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
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
        features = self.audio_encoder(audio)
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
        # Tokenize text
        tokens = self.text_tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        # Get text embeddings
        outputs = self.text_encoder(**tokens)
        features = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        return F.normalize(self.text_projection(features), dim=-1)

    def forward(
        self, audio: torch.Tensor, text: list[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing audio and text embeddings and their similarity.

        Args:
            audio: Audio tensor of shape (batch_size, time_steps)
            text: List of text strings of length batch_size

        Returns:
            Tuple of (audio_embeddings, text_embeddings, logits)
        """
        # Get normalized embeddings
        audio_embeddings = self.encode_audio(audio)
        text_embeddings = self.encode_text(text)

        # Compute similarity matrix
        logits = torch.matmul(audio_embeddings, text_embeddings.t()) / self.temperature

        return audio_embeddings, text_embeddings, logits

    def compute_loss(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between audio and text embeddings.

        Args:
            audio_embeddings: Audio embeddings tensor
            text_embeddings: Text embeddings tensor
            logits: Similarity matrix

        Returns:
            Contrastive loss value
        """
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=self.device)

        # Compute cross entropy loss in both directions
        loss_audio = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.t(), labels)

        return (loss_audio + loss_text) / 2
