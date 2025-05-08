"""
Audio augmentation utilities.

This module provides functions for audio data augmentation, including:
- add_noise: Add background noise to audio at specified signal-to-noise ratio (SNR)
- mixup: Mixup augmentation for audio samples

New in this revision (2025‑05‑07):
- AugmentationProcessor now supports mixup at the **batch** level via
  `apply_batch_augmentations`, leveraging `n_mixup` from `MixupAugment`.
- Helper `_select_random_pairs` added for clean pair sampling logic.
- Minor refactors to keep all audio tensors on the configured device and
  to always return tensors (leaving any np↔tensor conversion to the caller
  if needed).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  – kept for downstream ops that often rely on F
import torchaudio

from esp_data_temp.dataset import GSPath

# Ensure RunConfig is **not** imported at the top level.
# from representation_learning.configs import RunConfig  # DO NOT ADD

################################################################################
# Primitive augmentations
################################################################################


def add_noise(
    audio: torch.Tensor | np.ndarray,
    noise_dir: str | Sequence[str],
    *,
    snr_db_range: Tuple[float, float] = (-5.0, 20.0),
    sample_rate: int = 16000,
    max_noise_samples: int = 1000,
    max_window_sec: float = 10.0,
) -> torch.Tensor:
    """Add background noise at a random SNR by *streaming* a window from disk.

    * Loads **at most** ``max_window_sec`` seconds of noise to minimise I/O.
    * Works with local paths and ``gs://`` buckets (via :class:`GSPath`).
    * Keeps the original tensor/ndarray interface.

    Returns:
        torch.Tensor: The audio with added noise.

    Raises:
        FileNotFoundError: If the noise directory is not found.
        RuntimeError: If noise file inspection or loading fails.
    """

    # ------------------------------------------------------------------
    # Prepare input audio tensor
    # ------------------------------------------------------------------
    is_np = isinstance(audio, np.ndarray)
    audio_t = torch.from_numpy(audio) if is_np else audio  # shape (B, T) or (T,)
    if audio_t.ndim == 1:
        audio_t = audio_t.unsqueeze(0)
    audio_len = audio_t.shape[-1]

    # ------------------------------------------------------------------
    # Collect candidate noise files
    # ------------------------------------------------------------------
    noise_dirs = [noise_dir] if isinstance(noise_dir, str) else list(noise_dir)
    noise_paths: List[Path] = []

    for dir_str in noise_dirs:
        dir_path: Path | "GSPath" = (
            GSPath(dir_str) if dir_str.startswith("gs://") else Path(dir_str)
        )
        if not dir_path.exists():
            raise FileNotFoundError(f"Noise directory not found: {dir_str}")

        for ext in (".wav", ".mp3", ".flac", ".ogg"):
            noise_paths.extend(dir_path.glob(f"*{ext}"))
            if len(noise_paths) >= max_noise_samples:
                break
        if len(noise_paths) >= max_noise_samples:
            break

    if not noise_paths:
        return audio  # no‑op if no files

    noise_path = random.choice(noise_paths)

    # ------------------------------------------------------------------
    # Determine how many frames to read (avoid whole‑file load)
    # ------------------------------------------------------------------
    try:
        info = torchaudio.info(noise_path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to inspect noise file {noise_path}: {exc}") from exc

    noise_sr = info.sample_rate
    total_frames = info.num_frames

    target_frames = min(int(sample_rate * max_window_sec), audio_len)

    if total_frames <= target_frames:
        frame_offset = 0  # will read entire clip (short file)
        num_frames = total_frames
    else:
        frame_offset = random.randint(0, total_frames - target_frames)
        num_frames = target_frames

    try:
        noise_wav, _ = torchaudio.load(
            noise_path, frame_offset=frame_offset, num_frames=num_frames
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to load noise file {noise_path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Normalise channels, resample, length‑match, and mix
    # ------------------------------------------------------------------
    if noise_wav.shape[0] > 1:
        noise_wav = noise_wav.mean(dim=0, keepdim=True)

    if noise_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(noise_sr, sample_rate)
        noise_wav = resampler(noise_wav)

    # Tile or crop to match the audio length
    if noise_wav.shape[1] < audio_len:
        reps = int(np.ceil(audio_len / noise_wav.shape[1]))
        noise_wav = noise_wav.repeat(1, reps)
    if noise_wav.shape[1] > audio_len:
        start = random.randint(0, noise_wav.shape[1] - audio_len)
        noise_wav = noise_wav[:, start : start + audio_len]

    noise_wav = noise_wav.to(audio_t.device)
    audio_t = audio_t.to(noise_wav.device)

    # Scale noise to target SNR
    signal_power = (audio_t**2).mean(dim=1, keepdim=True)
    noise_power = (noise_wav**2).mean(dim=1, keepdim=True)

    snr_db = random.uniform(*snr_db_range)
    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-8))

    noisy = audio_t + scale * noise_wav
    return noisy.squeeze(0) if audio.ndim == 1 else noisy


def mixup(
    audio1: torch.Tensor | np.ndarray,
    audio2: torch.Tensor | np.ndarray,
    *,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, float]:
    """Return *mixed* audio and lambda sampled ~ Beta(alpha, alpha).

    Returns
    -------
    tuple[torch.Tensor, float]
        The mixed audio and the lambda value sampled from Beta(alpha, alpha).
    """
    is_np = isinstance(audio1, np.ndarray)
    x1 = torch.from_numpy(audio1) if is_np else audio1
    x2 = torch.from_numpy(audio2) if is_np else audio2

    lam = float(np.random.beta(alpha, alpha))
    mixed = lam * x1 + (1.0 - lam) * x2

    return mixed if not is_np else mixed.numpy(), lam


################################################################################
# Batch‑level augmentation engine
################################################################################


class AugmentationProcessor:
    """Applies audio augmentations (noise + mixup) according to config specs."""

    def __init__(
        self,
        augmentation_specs: Sequence[Any],
        sample_rate: int,
        device: str = "cpu",
    ) -> None:
        from representation_learning.configs import (  # local import
            MixupAugment,
            NoiseAugment,
        )

        self.device = torch.device(device)
        self.sr = sample_rate
        self.noise_aug_configs = [
            spec for spec in augmentation_specs if isinstance(spec, NoiseAugment)
        ]
        self.mixup_aug_configs = [
            spec for spec in augmentation_specs if isinstance(spec, MixupAugment)
        ]

    # ------------------------------------------------------------------
    # Per‑item processing (noise only)
    # ------------------------------------------------------------------
    def _apply_noise(self, wav: torch.Tensor) -> torch.Tensor:
        for cfg in self.noise_aug_configs:
            if random.random() < cfg.augmentation_prob:
                wav = add_noise(
                    wav,
                    noise_dir=cfg.noise_dirs,
                    snr_db_range=cfg.snr_db_range,
                    sample_rate=self.sr,
                )
                # add_noise already keeps type & device
        return wav

    def apply_augmentations(self, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Noise‑only path kept for backward compatibility.

        Parameters
        ----------
        item_dict : Dict[str, Any]
            A dictionary containing at least "raw_wav".

        Returns
        -------
        Dict[str, Any]
            The item dictionary with noise augmentation applied to "raw_wav".
        """
        aug_item = item_dict.copy()
        wav: torch.Tensor = item_dict["raw_wav"].to(self.device)
        aug_item["raw_wav"] = self._apply_noise(wav)
        return aug_item

    # ------------------------------------------------------------------
    # Batch processing – noise + optional mixup
    # ------------------------------------------------------------------
    def apply_batch_augmentations(
        self, batch: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply configured augmentations to an *entire* batch. Mostly for Mixup, others
        can go in the per‑item processing for parallel processing.

        Workflow:
        For each MixupAugment spec: with probability *augmentation_prob*,
           perform up to *n_mixup* random pairings within the batch.

        The mixed audio overwrites the *first* element in each pair and records
        metadata ("mixup_lambda", "mixup_partner_idx"). Labels are *not* mixed
        here because the downstream task‑specific collate_fn often implements
        that logic. Modify as required for your label schemata.

        Parameters
        ----------
        batch : Sequence[Dict[str, Any]]
            A sequence of item dictionaries, each containing at least "raw_wav".

        Returns
        -------
        List[Dict[str, Any]]
            The list of item dictionaries with augmentations applied.
        """
        if not batch:
            return []
        aug_batch = list(batch)  # Fix: define aug_batch from batch
        # 2. Mixup augmentation --------------------------------------
        if len(aug_batch) < 2 or not self.mixup_aug_configs:
            return aug_batch  # nothing to mix

        for cfg in self.mixup_aug_configs:
            if random.random() >= cfg.augmentation_prob:
                continue

            n_pairs = min(cfg.n_mixup, len(aug_batch) // 2)
            if n_pairs == 0:
                continue

            pairs = self._select_random_pairs(len(aug_batch), n_pairs)
            for i, j in pairs:
                mixed_wav, lam = mixup(
                    aug_batch[i]["raw_wav"], aug_batch[j]["raw_wav"], alpha=cfg.alpha
                )
                aug_batch[i]["raw_wav"] = mixed_wav  # type: ignore[assignment]
                aug_batch[i]["mixup_lambda"] = lam
                aug_batch[i]["mixup_partner_idx"] = j
                # Optionally record partner label etc.

        return aug_batch

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _select_random_pairs(batch_size: int, n_pairs: int) -> List[Tuple[int, int]]:
        """Return *n_pairs* disjoint index pairs randomly sampled from *batch_size*.

        Parameters
        ----------
        batch_size : int
            The total number of items to sample from.
        n_pairs : int
            The number of disjoint pairs to select.

        Returns
        -------
        List[Tuple[int, int]]
            A list of tuples, where each tuple contains two distinct indices.
        """
        indices = list(range(batch_size))
        random.shuffle(indices)
        pairs = [(indices[2 * k], indices[2 * k + 1]) for k in range(n_pairs)]
        return pairs
