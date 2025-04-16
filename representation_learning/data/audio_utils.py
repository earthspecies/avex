"""
Audio processing utilities for converting raw waveforms to various representations.
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import random
from torch import Tensor
from typing import Optional, Tuple, Literal, Union

from representation_learning.configs import AudioConfig

def pad_or_window(
    wav: np.ndarray,
    target_len: int,
    window_selection: str = "random"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure the waveform has exactly `target_len` samples.

    Returns
    -------
    windowed_wav : np.ndarray [target_len]          – either truncated or padded
    padding_mask : np.ndarray [target_len] bool     – True where audio is real

    Notes
    -----
    * **Longer** than target_len → choose a window of length `target_len`.
      - `mode="random"` picks a random start index.
      - Other modes can be added later (e.g. "center").
    * **Shorter** than target_len → pad **zeros** at the end.
    """
    if window_selection != "random":
            raise NotImplementedError(f"Window mode '{window_selection}' not implemented")
    
    wav_len = len(wav)

    if wav_len == target_len:
        mask = np.ones(target_len, dtype=bool)
        return wav.astype(np.float32), mask

    if wav_len > target_len:  # need to crop
        start = random.randint(0, wav_len - target_len)
        end   = start + target_len
        window = wav[start:end]
        mask   = np.ones(target_len, dtype=bool)
        return window.astype(np.float32), mask

    # wav_len < target_len  → pad zeros
    pad_len = target_len - wav_len
    padded  = np.pad(wav, (0, pad_len), mode="constant")
    mask    = np.zeros(target_len, dtype=bool)
    mask[:wav_len] = True
    return padded.astype(np.float32), mask

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
        self.target_length = cfg.target_length
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
        """
        Args
        ----
        waveform : (T,) or (B, T)
            Raw mono waveform in **float32** PCM (‑1 … 1).

        Returns
        -------
        Tensor
            • raw →  (B, T)  
            • spectrogram / mel_spectrogram →  (B, F, T')
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        # Ensure (B, T) shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Pad / crop to fixed length if requested
        if self.target_length is not None:
            wav_np, _ = pad_or_window(
                waveform.numpy(),
                target_len=self.target_length,
                window_selection=self.window_selection,
                sr=self.sr,
            )
            waveform = torch.from_numpy(wav_np)

        if self.representation == "raw":
            return waveform

        # STFT → power spectrogram
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
            x.amax(dim=(-2, -1), keepdim=True) - x.amin(dim=(-2, -1), keepdim=True) + 1e-8
        )
