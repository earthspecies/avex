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

def pad_or_window(
    wav: np.ndarray,
    target_len: int,
    window_selection: str = "random",
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
    """Processes raw audio waveforms into various representations."""
    
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Literal["hann", "hamming"] = "hann",
        n_mels: int = 128,
        representation: Literal["spectrogram", "mel_spectrogram", "raw"] = "mel_spectrogram",
        normalize: bool = True,
        target_length: Optional[int] = None,
        window_selection: str = "random",
    ):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: Number of FFT bins
            hop_length: Hop length between STFT windows
            win_length: Window length for STFT
            window: Window function type
            n_mels: Number of mel bands
            representation: Type of audio representation to use
            normalize: Whether to normalize the output
            target_length: Target length in samples for padding/windowing
            window_selection: Method for selecting windows ("random" or "center")
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        self.window = window
        self.n_mels = n_mels
        self.representation = representation
        self.normalize = normalize
        self.target_length = target_length
        self.window_selection = window_selection
        
        # Initialize mel basis if needed
        if representation == "mel_spectrogram":
            self.mel_basis = torchaudio.transforms.MelScale(
                n_mels=n_mels,
                sample_rate=sample_rate,
                n_stft=n_fft // 2 + 1
            )
    
    def __call__(self, waveform: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Convert raw waveform to the specified representation.
        
        Args:
            waveform: Input waveform tensor of shape (batch_size, time_steps)
            
        Returns:
            Processed audio representation tensor
        """
        # Convert numpy array to tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
            
        # Apply padding/windowing if target length is specified
        if self.target_length is not None:
            # Convert to numpy for pad_or_window
            wav_np = waveform.numpy()
            wav_np, _ = pad_or_window(wav_np, self.target_length, self.window_selection)
            waveform = torch.from_numpy(wav_np)
            
        if self.representation == "raw":
            return waveform
            
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._get_window(),
            return_complex=True
        )
        
        # Convert to power spectrogram
        spectrogram = torch.abs(stft) ** 2
        
        if self.representation == "spectrogram":
            output = spectrogram
        elif self.representation == "mel_spectrogram":
            output = self.mel_basis(spectrogram)
        else:
            raise ValueError(f"Unknown representation type: {self.representation}")
            
        # Normalize if requested
        if self.normalize:
            output = self._normalize(output)
            
        return output
    
    def _get_window(self) -> Tensor:
        """Get the window function tensor."""
        if self.window == "hann":
            return torch.hann_window(self.win_length)
        elif self.window == "hamming":
            return torch.hamming_window(self.win_length)
        else:
            raise ValueError(f"Unknown window type: {self.window}")
    
    def _normalize(self, x: Tensor) -> Tensor:
        """Normalize the spectrogram."""
        # Add small epsilon to avoid log(0)
        x = torch.log(x + 1e-6)
        
        # Normalize to [0, 1]
        x = (x - x.min()) / (x.max() - x.min())
        
        return x
