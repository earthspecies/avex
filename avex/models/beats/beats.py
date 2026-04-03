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
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from pydantic import BaseModel, ConfigDict, Field
from torch.nn import LayerNorm

from .backbone import TransformerEncoder

logger = logging.getLogger(__name__)

_FLOAT32_EPS = torch.finfo(torch.float32).eps


class _BatchedFbank(nn.Module):
    """GPU-native batched fbank that reproduces torchaudio.compliance.kaldi.fbank.

    Pre-computes the Povey window and mel filterbank matrix as registered
    buffers so they follow the module to whatever device the model lives on.
    All forward-pass operations are batched and differentiable.
    """

    def __init__(
        self,
        num_mel_bins: int = 128,
        sample_frequency: float = 16000.0,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        preemphasis_coefficient: float = 0.97,
        low_freq: float = 20.0,
        high_freq: float = 0.0,
    ) -> None:
        super().__init__()
        self.preemphasis_coefficient = preemphasis_coefficient

        win_length = int(sample_frequency * frame_length_ms / 1000.0)
        hop_length = int(sample_frequency * frame_shift_ms / 1000.0)
        self.win_length = win_length
        self.hop_length = hop_length

        n_fft = 1
        while n_fft < win_length:
            n_fft *= 2
        self.n_fft = n_fft
        self._pad_right = n_fft - win_length

        if high_freq <= 0.0:
            high_freq = sample_frequency / 2.0 + high_freq

        # Povey window: hann^0.85, matching kaldi's _feature_window_function
        window = torch.hann_window(win_length, periodic=False).pow(0.85)
        self.register_buffer("window", window)

        # Mel filterbank [n_fft//2 + 1, num_mel_bins] — matches get_mel_banks
        mel_fb = self._build_mel_filterbank(n_fft, num_mel_bins, sample_frequency, low_freq, high_freq)
        self.register_buffer("mel_fb", mel_fb)

    @staticmethod
    def _build_mel_filterbank(
        n_fft: int,
        n_mels: int,
        sample_rate: float,
        low_freq: float,
        high_freq: float,
    ) -> torch.Tensor:
        """Build triangular mel filterbank identical to kaldi's get_mel_banks.

        Returns:
            Tensor of shape [n_fft // 2 + 1, n_mels].
        """
        num_fft_bins = n_fft // 2
        fft_bin_width = sample_rate / n_fft

        mel_low = 1127.0 * math.log(1.0 + low_freq / 700.0)
        mel_high = 1127.0 * math.log(1.0 + high_freq / 700.0)
        mel_delta = (mel_high - mel_low) / (n_mels + 1)

        # [n_mels, 1]
        bin_idx = torch.arange(n_mels).unsqueeze(1)
        left_mel = mel_low + bin_idx * mel_delta
        center_mel = mel_low + (bin_idx + 1.0) * mel_delta
        right_mel = mel_low + (bin_idx + 2.0) * mel_delta

        # [1, num_fft_bins] — mel of each FFT bin (excluding Nyquist)
        freqs = fft_bin_width * torch.arange(num_fft_bins)
        mel_freqs = (1127.0 * (1.0 + freqs / 700.0).log()).unsqueeze(0)

        up_slope = (mel_freqs - left_mel) / (center_mel - left_mel)
        down_slope = (right_mel - mel_freqs) / (right_mel - center_mel)
        fb = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))

        # fb is [n_mels, num_fft_bins]; pad Nyquist column with 0, then transpose
        fb = F.pad(fb, (0, 1), value=0.0)  # [n_mels, num_fft_bins + 1]
        return fb.T  # [n_fft//2 + 1, n_mels]

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Compute log-mel filterbank features for a batch of waveforms.

        Reproduces the exact pipeline of ``torchaudio.compliance.kaldi.fbank``
        with default parameters (snip_edges=True, remove_dc_offset=True,
        preemphasis=0.97, povey window, use_power=True, use_log_fbank=True,
        dither=0.0) but operates on the full ``[B, T]`` batch at once.

        Args:
            waveforms: ``[B, T]`` tensor of raw audio (should already be
                       scaled by ``2**15`` if that is what kaldi expects).

        Returns:
            ``[B, num_frames, num_mel_bins]`` log-mel filterbank features.
        """
        # 1. Frame: snip_edges=True ⇔ center=False unfold
        frames = waveforms.unfold(-1, self.win_length, self.hop_length)
        # frames: [B, num_frames, win_length]

        # 2. Remove DC offset per frame
        frames = frames - frames.mean(dim=-1, keepdim=True)

        # 3. Pre-emphasis per frame (kaldi's replicate-pad approach)
        shifted = F.pad(frames, (1, 0), mode="replicate")[..., :-1]
        frames = frames - self.preemphasis_coefficient * shifted

        # 4. Apply Povey window
        frames = frames * self.window

        # 5. Zero-pad to n_fft
        if self._pad_right > 0:
            frames = F.pad(frames, (0, self._pad_right))

        # 6. Power spectrum: |FFT|^2
        spectrum = torch.fft.rfft(frames)
        power = spectrum.abs().pow(2.0)
        # power: [B, num_frames, n_fft // 2 + 1]

        # 7. Mel filterbank
        mel_energies = torch.matmul(power, self.mel_fb)
        # mel_energies: [B, num_frames, num_mel_bins]

        # 8. Log with float32-epsilon floor (matches kaldi)
        return torch.clamp(mel_energies, min=_FLOAT32_EPS).log()


class BEATsConfig(BaseModel):
    """Configuration for BEATs model parameters.

    This Pydantic model defines all configuration options for the BEATs
    (Bidirectional Encoder representation from Audio Transformers) architecture.
    Default values are set to match the iter3+AS2M fine-tuned variant.

    Example:
        >>> config = BEATsConfig()  # Use defaults
        >>> config = BEATsConfig(encoder_layers=6)  # Override specific fields
        >>> config = BEATsConfig(**{"encoder_layers": 6})  # Load from dict
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

    # Spectrogram / preprocessing
    sample_frequency: float = Field(16000.0, description="Audio sample rate in Hz")
    num_mel_bins: int = Field(128, description="Number of mel filterbank bins")
    frame_length: float = Field(25.0, description="Frame length in milliseconds")
    frame_shift: float = Field(10.0, description="Frame shift (hop) in milliseconds")
    fbank_mean: float = Field(15.41663, description="Mean for filterbank normalization")
    fbank_std: float = Field(6.55582, description="Standard deviation for filterbank normalization")

    # Label predictor (for fine-tuned models)
    finetuned_model: bool = Field(True, description="Whether this is a fine-tuned model")
    predictor_dropout: float = Field(0.0, description="Dropout probability for the predictor")
    predictor_class: int = Field(527, description="Target class number for the predictor")

    # Allow extra fields from checkpoints that may have additional config keys
    model_config = ConfigDict(extra="allow")


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

        self.fbank = _BatchedFbank(
            num_mel_bins=cfg.num_mel_bins,
            sample_frequency=cfg.sample_frequency,
            frame_length_ms=cfg.frame_length,
            frame_shift_ms=cfg.frame_shift,
        )
        self.fbank_mean = cfg.fbank_mean
        self.fbank_std = cfg.fbank_std

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

    def preprocess(self, source: torch.Tensor) -> torch.Tensor:
        """Preprocess audio waveforms to filterbank features.

        Uses the GPU-native batched fbank that reproduces kaldi's output,
        operating on the full batch in one shot instead of looping per sample.

        Args:
            source: ``[B, T]`` raw waveform tensor

        Returns:
            ``[B, num_frames, num_mel_bins]`` normalized filterbank features
        """
        fbank = self.fbank(source * 2**15)
        return (fbank - self.fbank_mean) / (2 * self.fbank_std)

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        feature_only: bool = False,
        disable_layerdrop: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Extract features from audio input.

        Args:
            source: Input audio tensor
            padding_mask: Optional padding mask
            feature_only: Whether to return only features (no predictions)
            disable_layerdrop: Whether to disable layerdrop during forward pass

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
                Features or tuple of (features/logits, padding_mask)
        """
        fbank = self.preprocess(source).to(torch.float32)

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
