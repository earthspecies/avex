"""
Audio processing utilities for converting raw waveforms to various representations.
"""

from typing import Literal, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor

from representation_learning.configs import AudioConfig


def pad_or_window(
    wav: torch.Tensor,
    target_len: int,
    window_selection: Literal["random", "center"] = "random",
    invert: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad or window a waveform to a target length.

    Parameters
    ----------
    wav : torch.Tensor
        Input waveform tensor
    target_len : int
        Target length to pad or window to
    window_selection : Literal["random", "center", "start"]
        How to select the window if cropping is needed
    invert : bool, default=True
        Whether to invert the boolean mask. When True, True values indicate padding
        regions.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of (processed waveform, mask)

    Raises
    ------
    ValueError
        If window_selection is not "random" or "center"
    """
    wav_len = wav.size(-1)
    mask = torch.ones(target_len, dtype=torch.bool)
    processed_wav = wav

    if wav_len == target_len:
        pass
    elif wav_len > target_len:  # crop
        if window_selection == "random":
            start = torch.randint(0, wav_len - target_len + 1, ()).item()
            end = start + target_len
            processed_wav = wav[..., start:end]
        elif window_selection == "center":
            start = (wav_len - target_len) // 2
            end = start + target_len
            processed_wav = wav[..., start:end]
        elif window_selection == "start":
            processed_wav = wav[..., :target_len]
        else:
            raise ValueError(f"Unknown window selection: {window_selection}")
    else:  # pad
        pad_len = target_len - wav_len
        processed_wav = F.pad(wav, (0, pad_len))
        mask[wav_len:] = False

    if invert:
        mask = ~mask

    return processed_wav, mask


class AudioProcessor:
    """Processes raw audio waveforms according to an `AudioConfig`."""

    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg

        # Convenience aliases
        self.sr = cfg.sample_rate
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length or self.n_fft // 4
        self.win_length = cfg.win_length or self.n_fft
        self.window_type = cfg.window
        self.n_mels = cfg.n_mels
        self.representation = cfg.representation
        self.normalize = cfg.normalize
        self.target_length_seconds = cfg.target_length_seconds
        self.window_selection = cfg.window_selection

        # Pre‑compute mel filter bank if required
        if self.representation == "mel_spectrogram":
            self.mel_basis = torchaudio.transforms.MelScale(
                n_mels=self.n_mels,
                sample_rate=self.sr,
                n_stft=self.n_fft // 2 + 1,
            )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def __call__(self, waveform: Union[Tensor, np.ndarray]) -> Tensor:
        """Process a waveform into the configured representation.

        Parameters
        ----------
        waveform : Union[Tensor, np.ndarray]
            Raw mono waveform in **float32** PCM (‑1 … 1).
            Shape: (T,) or (B, T)

        Returns
        -------
        Tensor
            • raw →  (B, T)
            • spectrogram / mel_spectrogram →  (B, F, T')

        Raises
        ------
        ValueError
            If the representation type is unknown
        """
        # Ensure (B, T) shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if self.representation == "raw":
            return waveform

        # Move mel basis to same device as input if needed
        if self.representation == "mel_spectrogram":
            self.mel_basis = self.mel_basis.to(waveform.device)

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._get_window().to(waveform.device),
            return_complex=True,
        )
        spectrogram = stft.abs().pow(2)

        if self.representation == "spectrogram":
            output = spectrogram
        elif self.representation == "mel_spectrogram":
            output = self.mel_basis(spectrogram)
        else:
            raise ValueError(f"Unknown representation: {self.representation}")

        return self._normalize(output) if self.normalize else output

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _get_window(self) -> Tensor:
        if self.window_type == "hann":
            return torch.hann_window(self.win_length)
        if self.window_type == "hamming":
            return torch.hamming_window(self.win_length)
        raise ValueError(f"Unknown window type: {self.window_type}")

    @staticmethod
    def _normalize(x: Tensor) -> Tensor:
        x = x = torch.log(x + 1e-6)
        return (x - x.amin(dim=(-2, -1), keepdim=True)) / (
            x.amax(dim=(-2, -1), keepdim=True)
            - x.amin(dim=(-2, -1), keepdim=True)
            + 1e-8
        )
