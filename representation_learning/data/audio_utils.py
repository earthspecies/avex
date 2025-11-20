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


def detect_active_regions(
    wav: torch.Tensor,
    sr: int,
    energy_threshold_db: float = -40.0,
    min_region_length_ms: int = 100,
    frame_length_ms: int = 25,
) -> list[tuple[int, int]]:
    """Detect active (non-silent) regions in audio using energy-based VAD.

    Parameters
    ----------
    wav : torch.Tensor
        Mono waveform tensor of shape (T,)
    sr : int
        Sample rate
    energy_threshold_db : float
        Energy threshold in dB relative to max energy
    min_region_length_ms : int
        Minimum length of active region to consider (ms)
    frame_length_ms : int
        Frame size for energy computation (ms)

    Returns
    -------
    list[tuple[int, int]]
        List of (start_sample, end_sample) tuples for active regions
    """
    # Convert to numpy for efficient frame-based processing
    wav_np = wav.cpu().numpy()
    frame_length_samples = int(frame_length_ms * sr / 1000)
    hop_length_samples = frame_length_samples // 2
    
    if len(wav_np) < frame_length_samples:
        return []
    
    # Compute RMS energy per frame
    frames = []
    for i in range(0, len(wav_np) - frame_length_samples + 1, hop_length_samples):
        frame = wav_np[i:i + frame_length_samples]
        rms = np.sqrt(np.mean(frame ** 2))
        frames.append((i, rms))
    
    if not frames:
        return []
    
    # Convert to dB and find threshold (using adaptive threshold based on max energy)
    max_rms = max(rms for _, rms in frames)
    if max_rms < 1e-10:  # Avoid division by zero for silent audio
        return []
    
    threshold_linear = max_rms * (10 ** (energy_threshold_db / 20))
    
    # Find active regions
    active_regions = []
    in_region = False
    region_start = 0
    
    for frame_start, rms in frames:
        is_active = rms > threshold_linear
        
        if is_active and not in_region:
            # Start of new region
            region_start = frame_start
            in_region = True
        elif not is_active and in_region:
            # End of region
            region_end = frame_start + frame_length_samples
            min_length_samples = int(min_region_length_ms * sr / 1000)
            if region_end - region_start >= min_length_samples:
                active_regions.append((region_start, region_end))
            in_region = False
    
    # Handle case where audio ends while in active region
    if in_region:
        region_end = len(wav_np)
        min_length_samples = int(min_region_length_ms * sr / 1000)
        if region_end - region_start >= min_length_samples:
            active_regions.append((region_start, region_end))
    
    return active_regions


def select_activity_window(
    wav: torch.Tensor,
    active_regions: list[tuple[int, int]],
    target_length_samples: int,
    min_window_length_samples: int,
) -> tuple[torch.Tensor, int, int] | None:
    """Select a random window from active regions.

    Parameters
    ----------
    wav : torch.Tensor
        Input waveform tensor
    active_regions : list[tuple[int, int]]
        List of (start_sample, end_sample) tuples for active regions
    target_length_samples : int
        Desired window length in samples
    min_window_length_samples : int
        Minimum window length in samples

    Returns
    -------
    tuple[torch.Tensor, int, int] | None
        (windowed_wav, start_idx, end_idx) or None if no suitable region found
    """
    if not active_regions:
        return None
    
    # Filter regions that are long enough
    valid_regions = [
        (start, end) for start, end in active_regions
        if (end - start) >= min_window_length_samples
    ]
    
    if not valid_regions:
        return None
    
    # Select random region
    region_start, region_end = valid_regions[
        torch.randint(0, len(valid_regions), ()).item()
    ]
    
    # Select random window within region
    region_length = region_end - region_start
    if region_length <= target_length_samples:
        # Use entire region (will pad later)
        window_start = int(region_start)
        window_end = int(region_end)
    else:
        # Random window within region
        max_start = int(region_start + region_length - target_length_samples)
        window_start = int(torch.randint(region_start, max_start + 1, ()).item())
        window_end = int(window_start + target_length_samples)
    
    windowed_wav = wav[window_start:window_end]
    
    return windowed_wav, window_start, window_end


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
        self.center = cfg.center

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
            center=self.center,
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

    @staticmethod
    def _normalize_zscore(x: Tensor) -> Tensor:
        x = torch.log(x + 1e-6)
        mean = x.mean(dim=-1, keepdim=True)  # (B, F, 1)
        std = x.std(dim=-1, keepdim=True).clamp(1e-5)
        return (x - mean) / std


# --------------------------------------------------------------------------- #
#  Padding-mask helpers (added for precise mask propagation to frame / patch) #
# --------------------------------------------------------------------------- #


def waveform_to_frame_mask(
    padding_mask: Tensor,
    *,
    hop_length: int,
) -> Tensor:
    """Down-sample a **sample-level** padding mask to STFT frame resolution.

    Parameters
    ----------
    padding_mask : Tensor
        Boolean mask shaped ``(B, T_samples)`` where *True* denotes **padded**
        (invalid) samples.
    hop_length : int
        Hop length (stride) used for the STFT.

    Returns
    -------
    Tensor
        Boolean mask of shape ``(B, T_frames)`` with the same semantics
        (*True* → frame should be masked).  A frame is marked as padded **only
        if *all* samples that map to it are padded**, mimicking the behaviour
        used by BEATs' ``forward_padding_mask`` helper.

    Raises
    ------
    ValueError
        If padding_mask does not have exactly 2 dimensions.
    """

    if padding_mask.ndim != 2:
        raise ValueError("Expected padding_mask of shape (B, T)")

    bsz, n_samples = padding_mask.shape
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")

    # Trim so the length is divisible by *hop_length* to allow cheap view-based
    # pooling (this replicates the logic used in BEATs).
    extra = n_samples % hop_length
    if extra > 0:
        padding_mask = padding_mask[:, :-extra]

    n_frames = padding_mask.shape[1] // hop_length
    frame_mask = padding_mask.view(bsz, n_frames, hop_length).all(dim=-1)
    return frame_mask


def sync_crop_or_pad_time(
    spec: Tensor,
    frame_mask: Tensor | None,
    target_len: int,
) -> tuple[Tensor, Tensor | None]:
    """Centre-crop or right-pad *spec* **and** *frame_mask* identically.

    This mirrors the behaviour of :py:meth:`Model._pad_or_crop_time` but keeps
    the two tensors in lock-step so their time dimensions always agree.

    Parameters
    ----------
    spec : Tensor
        Spectrogram of shape ``(B, T, F)``.
    frame_mask : Tensor | None
        Optional frame-level padding mask of shape ``(B, T)`` where *True*
        denotes padded frames.
    target_len : int
        Desired time dimension after cropping / padding.

    Returns
    -------
    tuple[Tensor, Tensor | None]
        ``(spec_out, mask_out)`` – *mask_out* is ``None`` when *frame_mask* was
        ``None``.

    Raises
    ------
    ValueError
        If spec does not have exactly 3 dimensions or if frame_mask dimensions
        don't match spec's batch and time dimensions.
    """

    bsz, t, feat_dim = spec.shape

    if t == target_len:
        if frame_mask is not None and frame_mask.shape[1] != target_len:
            raise ValueError("frame_mask length does not match spectrogram")
        return spec, frame_mask

    # --------------------------- crop ---------------------------------- #
    if t > target_len:
        start = (t - target_len) // 2
        spec_out = spec[:, start : start + target_len, :]
        if frame_mask is not None:
            mask_out = frame_mask[:, start : start + target_len]
        else:
            mask_out = None
        return spec_out, mask_out

    # --------------------------- pad  ---------------------------------- #
    pad_len = target_len - t
    pad_spec = spec.new_zeros(bsz, pad_len, feat_dim)
    spec_out = torch.cat([spec, pad_spec], dim=1)

    if frame_mask is not None:
        pad_mask = torch.ones(
            bsz, pad_len, dtype=frame_mask.dtype, device=frame_mask.device
        )
        mask_out = torch.cat([frame_mask, pad_mask], dim=1)
    else:
        mask_out = None

    return spec_out, mask_out


# --------------------------------------------------------------------------- #
#  Patch-level mask helper                                                   #
# --------------------------------------------------------------------------- #


def frame_mask_to_patch_mask(
    frame_mask: Tensor,
    *,
    patch_size_time: int,
    n_freq_bins: int,
) -> Tensor:
    """Convert a frame-level mask to the flattened 2-D patch sequence mask.

    The token ordering matches the `einsum('nchpwq -> nhwpqc')` pattern used in
    :pymeth:`ImageEncoder.patchify` (row-major across time patches, then
    frequency patches).

    Returns
    -------
    Tensor
        Boolean mask of shape ``(B, T_patches * F_patches)`` where *True*
        denotes masked patches.

    Raises
    ------
    ValueError
        If frame_mask does not have exactly 2 dimensions.
    """

    if frame_mask.ndim != 2:
        raise ValueError("Expected frame_mask of shape (B, T_frames)")

    bsz, t_frames = frame_mask.shape
    if t_frames % patch_size_time != 0:
        raise ValueError("Time dimension must be divisible by patch size")

    t_patches = t_frames // patch_size_time
    time_patch_mask = frame_mask.view(bsz, t_patches, patch_size_time).all(dim=-1)

    # Frequency dimension is always valid; replicate the time-patch mask across
    # *n_freq_patches* columns.
    freq_patches = n_freq_bins // patch_size_time
    patch_mask = time_patch_mask.repeat_interleave(freq_patches, dim=1)
    return patch_mask
