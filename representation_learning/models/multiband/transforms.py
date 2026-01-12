"""Multiband audio transforms via heterodyning.

This module provides transforms for splitting audio into frequency bands
via heterodyning (mixing to baseband), enabling processing of high sample
rate audio through models designed for lower sample rates.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF


@dataclass
class HeterodyneCfg:
    """Configuration for heterodyne transform."""

    baseband_sr: int = 16000
    lowpass_factor: float = 0.45  # cutoff = lowpass_factor * baseband_sr


class HeterodyneToBaseband(nn.Module):
    """Shifts a frequency band to baseband via heterodyning.

    The heterodyne process:
    1. Bandpass filter to isolate the target band
    2. Mix with a cosine at the band center frequency
    3. Lowpass filter to remove images
    4. Resample to baseband sample rate
    """

    def __init__(self, cfg: HeterodyneCfg):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        x: torch.Tensor,
        sr_in: int,
        f_low: float,
        f_high: float,
    ) -> Tuple[torch.Tensor, int]:
        """Apply heterodyne transform to shift band to baseband.

        Parameters
        ----------
        x : torch.Tensor
            Input waveform of shape (B, 1, T) or (B, T)
        sr_in : int
            Input sample rate
        f_low : float
            Lower frequency bound of band (Hz)
        f_high : float
            Upper frequency bound of band (Hz)

        Returns
        -------
        Tuple[torch.Tensor, int]
            Transformed waveform at baseband and output sample rate
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        device = x.device

        center = 0.5 * (f_low + f_high)
        bw = f_high - f_low
        if bw <= 0:
            raise ValueError("Bandwidth must be positive")

        Q = max(center / bw, 0.5)

        # Bandpass filter
        x_bp = torchaudio.functional.bandpass_biquad(
            waveform=x,
            sample_rate=sr_in,
            central_freq=center,
            Q=Q,
        )

        # Heterodyne (mix with carrier)
        t = torch.arange(T, device=device, dtype=x.dtype) / sr_in
        cos = torch.cos(2.0 * torch.pi * center * t).view(1, 1, T)
        x_mix = x_bp * cos

        # Lowpass filter (anti-alias before resampling)
        cutoff = self.cfg.lowpass_factor * self.cfg.baseband_sr
        x_lp = torchaudio.functional.lowpass_biquad(
            x_mix,
            sample_rate=sr_in,
            cutoff_freq=cutoff,
        )

        # Resample to baseband
        if sr_in != self.cfg.baseband_sr:
            x_out = torchaudio.functional.resample(
                x_lp,
                orig_freq=sr_in,
                new_freq=self.cfg.baseband_sr,
            )
        else:
            x_out = x_lp

        return x_out, self.cfg.baseband_sr


class MultibandTransform(nn.Module):
    """Splits audio into frequency bands via heterodyning.

    Given an input waveform, this transform:
    1. Computes valid frequency bands up to the Nyquist frequency
    2. Applies heterodyne transform to each band
    3. Returns stacked bands as (N, num_bands, T_baseband)

    The number of bands is determined dynamically based on the input
    sample rate and band configuration.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        baseband_sr: int = 16000,
        band_width_hz: int = 8000,
        step_hz: Optional[int] = None,
        lowpass_factor: float = 0.45,
    ):
        """Initialize multiband transform.

        Parameters
        ----------
        sample_rate : int
            Expected input sample rate
        baseband_sr : int
            Output sample rate for each band
        band_width_hz : int
            Width of each frequency band in Hz
        step_hz : int, optional
            Step size between bands. Defaults to band_width_hz (non-overlapping)
        lowpass_factor : float
            Lowpass cutoff as fraction of baseband_sr
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.baseband_sr = baseband_sr
        self.band_width_hz = band_width_hz
        self.step_hz = step_hz or band_width_hz
        self.lowpass_factor = lowpass_factor

        self.heterodyne = HeterodyneToBaseband(
            HeterodyneCfg(baseband_sr=baseband_sr, lowpass_factor=lowpass_factor)
        )

        # Compute valid bands
        self.bands = self._compute_bands(sample_rate)

    def _compute_bands(self, sample_rate: int) -> List[Tuple[float, float]]:
        """Compute valid frequency bands up to Nyquist."""
        nyquist = sample_rate / 2.0
        bands = []
        f = 0.0
        while f < nyquist:
            f_high = min(f + self.band_width_hz, nyquist)
            bands.append((f, f_high))
            f += self.step_hz
        return bands

    @property
    def num_bands(self) -> int:
        """Number of frequency bands."""
        return len(self.bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split audio into frequency bands.

        Parameters
        ----------
        x : torch.Tensor
            Input waveform of shape (N, T) or (N, 1, T)

        Returns
        -------
        torch.Tensor
            Multiband output of shape (N, num_bands, T_baseband)
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)

        outputs = []
        for f_low, f_high in self.bands:
            x_band, _ = self.heterodyne(x, self.sample_rate, f_low, f_high)
            outputs.append(x_band)

        # Pad to same length if needed (can vary slightly due to resampling)
        max_len = max(b.shape[-1] for b in outputs)
        outputs = [F.pad(b, (0, max_len - b.shape[-1])) for b in outputs]

        return torch.cat(outputs, dim=1)  # (N, num_bands, T_baseband)

    def get_band_info(self) -> List[Tuple[float, float]]:
        """Return list of (f_low, f_high) for each band."""
        return self.bands.copy()
