"""BEATs: Audio Pre-Training with Acoustic Tokenizers.

This module provides the BEATs
Bidirectional Encoder representation from Audio Transformers
model implementation for audio representation learning tasks.

Based on:
- Paper: https://arxiv.org/abs/2212.09058
- Original implementation: https://github.com/microsoft/unilm/tree/master/beats
- Copyright (c) 2022 Microsoft, Licensed under The MIT License
"""

# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as ta_kaldi
from pydantic import BaseModel, ConfigDict, Field
from torch.nn import LayerNorm

from .backbone import TransformerEncoder

logger = logging.getLogger(__name__)


class BEATsConfig(BaseModel):
    """Configuration for BEATs model parameters.

    This Pydantic model defines all configuration options for the BEATs
    (Bidirectional Encoder representation from Audio Transformers) architecture.
    Default values are set to match the iter3+AS2M fine-tuned variant.

    Example:
        >>> config = BEATsConfig()  # Use defaults
        >>> config = BEATsConfig(encoder_layers=6)  # Override specific fields
        >>> config = BEATsConfig.from_dict({"encoder_layers": 6})  # Load from dict
    """

    # Patch embedding configuration
    input_patch_size: int = Field(16, description="Patch size of patch embedding")
    embed_dim: int = Field(512, description="Patch embedding dimension")
    conv_bias: bool = Field(False, description="Include bias in conv encoder")

    # Encoder architecture
    encoder_layers: int = Field(12, description="Number of encoder layers in the transformer")
    encoder_embed_dim: int = Field(768, description="Encoder embedding dimension")
    encoder_ffn_embed_dim: int = Field(3072, description="Encoder FFN embedding dimension")
    encoder_attention_heads: int = Field(12, description="Number of encoder attention heads")
    activation_fn: str = Field("gelu", description="Activation function to use")

    # Training dynamics
    layer_wise_gradient_decay_ratio: float = Field(1.0, description="Ratio for layer-wise gradient decay")
    layer_norm_first: bool = Field(False, description="Apply layernorm first in the transformer")
    deep_norm: bool = Field(False, description="Apply deep_norm first in the transformer")

    # Dropout configuration
    dropout: float = Field(0.1, description="Dropout probability for the transformer")
    attention_dropout: float = Field(0.1, description="Dropout probability for attention weights")
    activation_dropout: float = Field(0.0, description="Dropout probability after activation in FFN")
    encoder_layerdrop: float = Field(0.05, description="Probability of dropping a transformer layer")
    dropout_input: float = Field(0.0, description="Dropout to apply to the input after feature extraction")

    # Positional embeddings
    conv_pos: int = Field(128, description="Number of filters for convolutional positional embeddings")
    conv_pos_groups: int = Field(16, description="Number of groups for convolutional positional embedding")

    # Relative position embedding
    relative_position_embedding: bool = Field(True, description="Apply relative position embedding")
    num_buckets: int = Field(320, description="Number of buckets for relative position embedding")
    max_distance: int = Field(800, description="Maximum distance for relative position embedding")
    gru_rel_pos: bool = Field(True, description="Apply gated relative position embedding")

    # Label predictor (for fine-tuned models)
    finetuned_model: bool = Field(True, description="Whether this is a fine-tuned model")
    predictor_dropout: float = Field(0.0, description="Dropout probability for the predictor")
    predictor_class: int = Field(527, description="Target class number for the predictor")

    # Allow extra fields from checkpoints that may have additional config keys
    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "BEATsConfig":
        """Create a BEATsConfig from a dictionary.

        This method provides backward compatibility for loading configurations
        from checkpoint files that store config as a dictionary.

        Args:
            cfg: Dictionary containing configuration parameters

        Returns:
            BEATsConfig: Validated configuration object
        """
        return cls(**cfg)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return self.model_dump()


class BEATs(nn.Module):
    """BEATs (Bidirectional Encoder representation from Audio Transformers) model."""

    def __init__(
        self,
        cfg: BEATsConfig,
    ) -> None:
        """Initialize BEATs model.

        Args:
            cfg: BEATs configuration object
        """
        super().__init__()
        logger.info(f"BEATs Config: {cfg.model_dump()}")

        self.cfg = cfg

        self.embed = cfg.embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim) if self.embed != cfg.encoder_embed_dim else None
        )

        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(
            1,
            self.embed,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=cfg.conv_bias,
        )

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Process padding mask to match feature dimensions.

        Args:
            features: Feature tensor
            padding_mask: Padding mask tensor

        Returns:
            torch.Tensor: Processed padding mask
        """
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def preprocess(
        self,
        source: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        """Preprocess audio waveforms to filterbank features.

        Args:
            source: Input waveform tensor
            fbank_mean: Mean for filterbank normalization
            fbank_std: Standard deviation for filterbank normalization

        Returns:
            torch.Tensor: Preprocessed filterbank features
        """
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2**15
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
        feature_only: bool = False,
        disable_layerdrop: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Extract features from audio input.

        Args:
            source: Input audio tensor
            padding_mask: Optional padding mask
            fbank_mean: Mean for filterbank normalization
            fbank_std: Standard deviation for filterbank normalization
            feature_only: Whether to return only features (no predictions)
            disable_layerdrop: Whether to disable layerdrop during forward pass

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
                Features or tuple of (features/logits, padding_mask)
        """
        fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std).to(torch.float32)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            disable_layerdrop=disable_layerdrop,
        )

        if not feature_only and self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)

            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
            else:
                logits = logits.mean(dim=1)

            return logits, padding_mask

        return x, padding_mask

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        disable_layerdrop: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward pass of BEATs model.

        Args:
            source: Input audio tensor
            padding_mask: Optional padding mask
            disable_layerdrop: Whether to disable layerdrop during forward pass

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
                Model output (features or predictions)
        """
        return self.extract_features(source, padding_mask, feature_only=True, disable_layerdrop=disable_layerdrop)
