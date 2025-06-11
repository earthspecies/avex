"""Audio augmentation utilities.

This module provides functions for audio data augmentation, including:
- AugmentationProcessor: Unified processor for noise and mixup augmentations
- mixup: Mixup augmentation for audio samples

Supports both item-level (noise) and batch-level (mixup) augmentation.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from esp_data_temp.dataset import GSPath
from representation_learning.data.data_utils import combine_text_labels

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure INFO logs are visible

# Constants
MIN_BATCH_SIZE_FOR_MIXUP = 2


def mixup(
    audio1: torch.Tensor | np.ndarray,
    audio2: torch.Tensor | np.ndarray,
    *,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, float]:
    """Return *mixed* audio and lambda sampled ~ Beta(alpha, alpha).

    Parameters
    ----------
    audio1 : torch.Tensor | np.ndarray
        First audio sample to mix.
    audio2 : torch.Tensor | np.ndarray
        Second audio sample to mix.
    alpha : float, default=0.2
        Beta distribution parameter for mixup lambda.

    Returns
    -------
    tuple[torch.Tensor, float]
        The mixed audio and the lambda value sampled from Beta(alpha, alpha).

    """
    is_np = isinstance(audio1, np.ndarray)
    x1 = torch.from_numpy(audio1) if is_np else audio1
    x2 = torch.from_numpy(audio2) if is_np else audio2

    rng = np.random.default_rng()
    lam = float(rng.beta(alpha, alpha))
    mixed = lam * x1 + (1.0 - lam) * x2

    return mixed if not is_np else mixed.numpy(), lam


################################################################################
# Unified augmentation processor
################################################################################


class AugmentationProcessor:
    """Unified processor for audio augmentations (noise + mixup)."""

    def __init__(
        self,
        augmentation_specs: Sequence[Any],
        sample_rate: int,
        device: str = "cpu",
    ) -> None:
        """Initialize the augmentation processor.

        Parameters
        ----------
        augmentation_specs : Sequence[Any]
            List of augmentation configuration objects.
        sample_rate : int
            Audio sample rate.
        device : str, default="cpu"
            Device to use for processing.

        Raises
        ------
        RuntimeError
            If noise file collection fails for any configured noise directory.

        """
        from representation_learning.configs import (  # noqa: PLC0415
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

        # Pre-collect noise file lists to avoid repeated directory scanning
        self._noise_pools: dict[int, list[Path]] = {}
        for cfg in self.noise_aug_configs:
            try:
                self._noise_pools[id(cfg)] = self._list_noise_files(cfg.noise_dirs)
            except Exception as exc:
                msg = (
                    f"Failed to build noise file list for dirs {cfg.noise_dirs}: {exc}"
                )
                raise RuntimeError(msg) from exc

        # Track recently used files for cache optimization
        self._recent_noise_files = []
        self._max_recent_files = 10

    @staticmethod
    def _list_noise_files(
        noise_dirs: Sequence[str],
        max_noise_samples: int = 10000,
    ) -> list[Path]:
        """Enumerate all candidate noise files across noise directories.

        Parameters
        ----------
        noise_dirs : Sequence[str]
            List of directory paths containing noise files.
        max_noise_samples : int, default=10000
            Maximum number of noise samples to collect per directory.

        Returns
        -------
        list[Path]
            List of noise file paths found in the directories.

        Raises
        ------
        FileNotFoundError
            If any of the specified noise directories do not exist.

        """
        noise_paths: list[Path] = []
        for dir_str in noise_dirs:
            dir_path: Path | GSPath = (
                GSPath(dir_str) if dir_str.startswith("gs://") else Path(dir_str)
            )
            if not dir_path.exists():
                msg = f"Noise directory not found: {dir_str}"
                raise FileNotFoundError(msg)

            for ext in (".wav", ".mp3", ".flac", ".ogg"):
                noise_files = list(dir_path.glob(f"*{ext}"))[:max_noise_samples]
                noise_paths.extend(noise_files)
        logger.info("Found %d noise files", len(noise_paths))
        return noise_paths

    # ------------------------------------------------------------------
    # Item-level noise augmentation
    # ------------------------------------------------------------------
    def _apply_noise(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply noise augmentation to a single audio sample.

        Parameters
        ----------
        wav : torch.Tensor
            Input audio tensor to augment with noise.

        Returns
        -------
        torch.Tensor
            Audio tensor with noise augmentation applied.

        """
        for cfg in self.noise_aug_configs:
            if random.random() >= cfg.augmentation_prob:  # noqa: S311
                continue

            noise_candidates = self._noise_pools.get(id(cfg))
            if not noise_candidates:
                continue

            noise_path = random.choice(noise_candidates)  # noqa: S311
            # Apply noise mixing
            wav = self._mix_noise(wav, noise_path, cfg.snr_db_range)

        return wav

    def _load_noise_segment(
        self,
        noise_path: Path | GSPath,
        audio_len: int,
        max_window_sec: float,
    ) -> torch.Tensor:
        """Load and process a noise segment.

        Parameters
        ----------
        noise_path : Path | GSPath
            Path to noise file.
        audio_len : int
            Target audio length in samples.
        max_window_sec : float
            Maximum window size in seconds.

        Returns
        -------
        torch.Tensor
            Processed noise audio tensor.

        """
        info = sf.info(str(noise_path))
        target_frames = min(int(self.sr * max_window_sec), audio_len)

        if info.frames <= target_frames:
            frame_offset, num_frames = 0, info.frames
        else:
            frame_offset = random.randint(0, info.frames - target_frames)  # noqa: S311
            num_frames = target_frames

        noise_wav, noise_sr = torchaudio.load(
            noise_path,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )

        # Convert to mono
        if noise_wav.shape[0] > 1:
            noise_wav = noise_wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if noise_sr != self.sr:
            resampler = torchaudio.transforms.Resample(noise_sr, self.sr)
            noise_wav = resampler(noise_wav)

        return noise_wav

    @staticmethod
    def _match_audio_length(
        noise_wav: torch.Tensor,
        audio_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Match noise audio length to target audio length.

        Parameters
        ----------
        noise_wav : torch.Tensor
            Input noise audio tensor.
        audio_len : int
            Target audio length in samples.
        device : torch.device
            Target device for the tensor.

        Returns
        -------
        torch.Tensor
            Noise audio tensor matched to target length.

        """
        noise_wav = noise_wav.to(device)

        if noise_wav.shape[-1] < audio_len:
            reps = int(np.ceil(audio_len / noise_wav.shape[-1]))
            noise_wav = noise_wav.repeat(1, reps)

        if noise_wav.shape[-1] > audio_len:
            start = random.randint(0, noise_wav.shape[-1] - audio_len)  # noqa: S311
            noise_wav = noise_wav[:, start : start + audio_len]

        return noise_wav

    def _mix_noise(
        self,
        wav: torch.Tensor,
        noise_path: Path | GSPath,
        snr_db_range: tuple[float, float],
        max_window_sec: float = 10.0,
    ) -> torch.Tensor:
        """Mix noise into audio signal at random SNR.

        Parameters
        ----------
        wav : torch.Tensor
            Input audio tensor.
        noise_path : Path | GSPath
            Path to noise file.
        snr_db_range : tuple[float, float]
            Range of SNR values in dB to randomly sample from.
        max_window_sec : float, default=10.0
            Maximum window size in seconds for noise loading.

        Returns
        -------
        torch.Tensor
            Audio tensor with noise mixed in.

        """
        audio_len = wav.shape[-1]
        audio_t = wav.unsqueeze(0) if wav.ndim == 1 else wav

        # Load and process noise
        noise_wav = self._load_noise_segment(noise_path, audio_len, max_window_sec)
        noise_wav = self._match_audio_length(noise_wav, audio_len, audio_t.device)

        # Apply SNR scaling
        signal_power = (audio_t**2).mean(dim=-1, keepdim=True)
        noise_power = (noise_wav**2).mean(dim=-1, keepdim=True)

        snr_db = random.uniform(*snr_db_range)  # noqa: S311
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-8))

        mixed = audio_t + scale * noise_wav
        return mixed.squeeze(0) if wav.ndim == 1 else mixed

    def apply_augmentations(self, item_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply item-level augmentations (noise only).

        Parameters
        ----------
        item_dict : dict[str, Any]
            Dictionary containing audio data and metadata.

        Returns
        -------
        dict[str, Any]
            Dictionary with augmented audio data.

        """
        aug_item = item_dict.copy()
        wav: torch.Tensor = item_dict["raw_wav"].to(self.device)
        aug_item["raw_wav"] = self._apply_noise(wav)
        return aug_item

    # ------------------------------------------------------------------
    # Batch-level augmentation (mixup)
    # ------------------------------------------------------------------
    def apply_batch_augmentations(
        self,
        batch: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply batch-level augmentations (mixup).

        Parameters
        ----------
        batch : Sequence[dict[str, Any]]
            Input batch of data items.

        Returns
        -------
        list[dict[str, Any]]
            Batch with mixup augmentations applied.

        """
        if not batch:
            return []

        aug_batch = list(batch)

        # Apply mixup if configured and batch size allows
        if len(aug_batch) < MIN_BATCH_SIZE_FOR_MIXUP or not self.mixup_aug_configs:
            return aug_batch

        for cfg in self.mixup_aug_configs:
            if random.random() >= cfg.augmentation_prob:  # noqa: S311
                continue

            n_pairs = random.randint(1, min(cfg.n_mixup, len(aug_batch) // 2))  # noqa: S311
            if n_pairs == 0:
                continue

            pairs = self._select_random_pairs(len(aug_batch), n_pairs)
            for i, j in pairs:
                mixed_wav, lam = mixup(
                    aug_batch[i]["raw_wav"],
                    aug_batch[j]["raw_wav"],
                    alpha=cfg.alpha,
                )
                aug_batch[i]["raw_wav"] = mixed_wav
                aug_batch[i]["mixup_lambda"] = lam
                aug_batch[i]["mixup_partner_idx"] = j

                # Combine labels for multi-label support
                if "label" in aug_batch[i] and "label" in aug_batch[j]:
                    lbl_i, lbl_j = aug_batch[i]["label"], aug_batch[j]["label"]

                    if not isinstance(lbl_i, (int, np.integer)) or not isinstance(
                        lbl_j,
                        (int, np.integer),
                    ):
                        # Multi-hot vectors: logical OR
                        tensor_i = (
                            torch.as_tensor(lbl_i, dtype=torch.float32)
                            if not isinstance(lbl_i, torch.Tensor)
                            else lbl_i.float()
                        )
                        tensor_j = (
                            torch.as_tensor(lbl_j, dtype=torch.float32)
                            if not isinstance(lbl_j, torch.Tensor)
                            else lbl_j.float()
                        )
                        aug_batch[i]["label"] = torch.maximum(tensor_i, tensor_j)

                # Combine text labels for CLIP training
                if "text_label" in aug_batch[i] and "text_label" in aug_batch[j]:
                    aug_batch[i]["text_label"] = combine_text_labels(
                        aug_batch[i]["text_label"],
                        aug_batch[j]["text_label"],
                    )

        return aug_batch

    @staticmethod
    def _select_random_pairs(batch_size: int, n_pairs: int) -> list[tuple[int, int]]:
        """Select n_pairs disjoint index pairs randomly.

        Parameters
        ----------
        batch_size : int
            Size of the batch to select pairs from.
        n_pairs : int
            Number of pairs to select.

        Returns
        -------
        list[tuple[int, int]]
            List of index pairs for mixup.

        """
        indices = list(range(batch_size))
        random.shuffle(indices)
        return [(indices[2 * k], indices[2 * k + 1]) for k in range(n_pairs)]


################################################################################
# Utility classes and functions
################################################################################


class ItemPostprocessor:
    """Wrapper for AugmentationProcessor for use as dataset postprocessor."""

    def __init__(self, aug_processor: AugmentationProcessor) -> None:
        """Initialize with an augmentation processor.

        Parameters
        ----------
        aug_processor : AugmentationProcessor
            The processor to wrap for dataset postprocessing.

        """
        self.aug_processor = aug_processor

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        """Apply augmentations with tensor conversion handling.

        Parameters
        ----------
        item : dict[str, Any]
            Input data item to process.

        Returns
        -------
        dict[str, Any]
            Processed data item with augmentations applied.

        """
        if not isinstance(item["raw_wav"], torch.Tensor):
            item["raw_wav"] = torch.from_numpy(item["raw_wav"])
        item = self.aug_processor.apply_augmentations(item)
        if isinstance(item["raw_wav"], torch.Tensor):
            item["raw_wav"] = item["raw_wav"].cpu().numpy()
        return item


def make_item_postprocessor(
    aug_processor: AugmentationProcessor,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a postprocessor function for dataset augmentation.

    Parameters
    ----------
    aug_processor : AugmentationProcessor
        The augmentation processor to use.

    Returns
    -------
    Callable[[dict[str, Any]], dict[str, Any]]
        A callable that can be used as a dataset postprocessor.

    """
    return ItemPostprocessor(aug_processor)
