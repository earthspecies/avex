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

import logging
import os
import random
import time
from collections import defaultdict
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  – kept for downstream ops that often rely on F
import torchaudio

from esp_data_temp.dataset import GSPath
from representation_learning.data.data_utils import combine_text_labels

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
        # Pre-collect noise file lists to avoid repeated directory scanning
        # ------------------------------------------------------------------
        self._noise_pools: Dict[int, List[Path]] = {}
        for cfg in self.noise_aug_configs:
            try:
                self._noise_pools[id(cfg)] = _list_noise_files(cfg.noise_dirs)
            except Exception as exc:  # pragma: no cover – configuration issue
                raise RuntimeError(
                    f"Failed to build noise file list for dirs {cfg.noise_dirs}: {exc}"
                ) from exc

        # Track recently used files to optimize future noise selections
        self._recent_noise_files = []
        self._max_recent_files = 10

    def prefetch_metadata(self, max_files_per_config: int = 100) -> None:
        """Prefetch audio file metadata to cache.

        Parameters
        ----------
        max_files_per_config : int, optional
            Maximum number of files to prefetch for each noise config, by default 100
        """
        # No need to log in this method - we only use timers for debugging

        prefetch_count = 0
        for _cfg_id, noise_files in self._noise_pools.items():
            # Prefetch a subset of files
            files_to_prefetch = min(max_files_per_config, len(noise_files))
            for noise_path in noise_files[:files_to_prefetch]:
                try:
                    _cached_audio_info(str(noise_path))
                    prefetch_count += 1
                except Exception as e:
                    logger.warning(f"Failed to prefetch info for {noise_path}: {e}")

        logger.info(f"Prefetched metadata for {prefetch_count} noise files")

    # ------------------------------------------------------------------
    # Per‑item processing (noise only)
    # ------------------------------------------------------------------
    def _apply_noise(self, wav: torch.Tensor) -> torch.Tensor:
        total_start = time.time()

        # Skip time tracking if no augmentation
        skip_augmentation = True
        for cfg in self.noise_aug_configs:
            if random.random() >= cfg.augmentation_prob:
                continue
            skip_augmentation = False
            break

        if skip_augmentation:
            return wav

        # We're going to apply noise, so time it
        for cfg in self.noise_aug_configs:
            if random.random() >= cfg.augmentation_prob:
                continue

            path_select_start = time.time()
            noise_candidates = self._noise_pools.get(id(cfg))
            if not noise_candidates:
                continue  # should not happen – safeguard

            # Prioritize selecting from recently used files if they exist
            # (better cache locality)
            if (
                self._recent_noise_files and random.random() < 0.3
            ):  # 30% chance to reuse recent files
                # Filter recent files that belong to the current config's pool
                recent_in_pool = [
                    f for f in self._recent_noise_files if f in noise_candidates
                ]
                if recent_in_pool:
                    noise_path = random.choice(recent_in_pool)
                else:
                    noise_path = random.choice(noise_candidates)
            else:
                noise_path = random.choice(noise_candidates)

            # Update recently used files list
            if noise_path in self._recent_noise_files:
                self._recent_noise_files.remove(noise_path)
            self._recent_noise_files.insert(0, noise_path)  # Add to front
            if len(self._recent_noise_files) > self._max_recent_files:
                self._recent_noise_files.pop()  # Remove oldest

            # Track timing but don't store if we're not logging
            _ = time.time() - path_select_start  # path_select_time

            # Time the actual noise mixing
            mix_start = time.time()
            wav = self._mix_in_noise(
                wav=wav,
                noise_path=noise_path,
                snr_db_range=cfg.snr_db_range,
            )
            _ = time.time() - mix_start  # mix_time

        _ = time.time() - total_start  # total_time

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

            n_pairs = random.randint(1, min(cfg.n_mixup, len(aug_batch) // 2))
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
                # --------------------------------------------------------
                # Combine *numeric* labels if present (multi-label support)
                # --------------------------------------------------------
                if "label" in aug_batch[i] and "label" in aug_batch[j]:
                    lbl_i = aug_batch[i]["label"]
                    lbl_j = aug_batch[j]["label"]

                    # Case 1: multi-hot vectors (array / tensor / list)
                    if not isinstance(lbl_i, (int, np.integer)) or not isinstance(
                        lbl_j, (int, np.integer)
                    ):
                        # Convert to float tensors for logical OR / max
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
                        # Union: mark a class as present if present in *either* parent
                        new_lbl = torch.maximum(tensor_i, tensor_j)
                        aug_batch[i]["label"] = new_lbl
                    # Case 2: single integer label – keep original (no mixing)
                if "text_label" in aug_batch[i] and "text_label" in aug_batch[j]:
                    aug_batch[i]["text_label"] = combine_text_labels(
                        aug_batch[i]["text_label"], aug_batch[j]["text_label"]
                    )

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

    # ------------------------------------------------------------------
    # Internal noise-mixing (file path already chosen, no directory scan!)
    # ------------------------------------------------------------------

    def _mix_in_noise(
        self,
        *,
        wav: torch.Tensor,
        noise_path: Path | "GSPath",
        snr_db_range: Tuple[float, float],
        max_window_sec: float = 10.0,
    ) -> torch.Tensor:
        """Overlay *wav* with noise from *noise_path* at random SNR.

        Returns
        -------
        torch.Tensor
            The audio with the noise mixed in
        """
        is_gcs_path = str(noise_path).startswith("gs://")

        # Only create timing variables if profiling is enabled
        if ENABLE_PROFILING:
            step_times = {}
            start_time = time.time()
            logger.info(
                f"[NOISE] Starting to mix {os.path.basename(str(noise_path))} | "
                f"GCS: {is_gcs_path}"
            )

        # Resolve audio length & target frames
        audio_len = wav.shape[-1]

        # --- Fast path: cached metadata – avoids I/O on every call --------
        if ENABLE_PROFILING:
            info_start = time.time()

        # Get audio file info with timing
        info = _cached_audio_info(str(noise_path))

        if ENABLE_PROFILING:
            metadata_time = time.time() - info_start
            # Compute cache status text first to avoid very long f-strings
            cache_status = (
                hasattr(_cached_audio_info, "cache_info")
                and _cached_audio_info.cache_info().hits > 0
            )
            logger.info(
                f"[NOISE] Metadata fetch time: {metadata_time:.4f}s | "
                f"Cache hit: {cache_status}"
            )

        # Don't need to store noise_sr, only used for resampling
        _ = info.sample_rate  # noise_sr

        total_frames = info.num_frames

        target_frames = min(int(self.sr * max_window_sec), audio_len)

        if total_frames <= target_frames:
            frame_offset = 0
            num_frames = total_frames
        else:
            frame_offset = random.randint(0, total_frames - target_frames)
            num_frames = target_frames

        # Measure time for loading audio
        if ENABLE_PROFILING:
            load_start = time.time()

        # Only log before loading if verbose debugging is enabled
        if os.environ.get("VERBOSE_NOISE_DEBUG", "0") == "1":
            logger.info(
                f"[NOISE-DETAIL] Loading from {os.path.basename(str(noise_path))} | "
                f"frames:{num_frames}, offset:{frame_offset}"
            )

        noise_wav, _sr = torchaudio.load(
            noise_path, frame_offset=frame_offset, num_frames=num_frames
        )

        # Only log after loading if verbose debugging is enabled
        if os.environ.get("VERBOSE_NOISE_DEBUG", "0") == "1":
            logger.info(
                f"[NOISE-DETAIL] Finished loading audio from "
                f"{os.path.basename(str(noise_path))}"
            )

        if ENABLE_PROFILING:
            load_time = time.time() - load_start
            # Only log if loading is slow (>0.05s)
            if load_time > 0.05:
                logger.info(
                    f"[NOISE] Audio load time: {load_time:.4f}s | GCS: {is_gcs_path} | "
                    f"File: {os.path.basename(str(noise_path))}"
                )

        # ------------------------------------------------------------------
        # Basic normalisation → mono → resample → length-match
        # ------------------------------------------------------------------
        if ENABLE_PROFILING:
            step_times = {}  # Initialize step_times dictionary here

        if noise_wav.shape[0] > 1:
            noise_wav = noise_wav.mean(dim=0, keepdim=True)

        if _sr != self.sr:
            if ENABLE_PROFILING:
                resample_start = time.time()

            resampler = torchaudio.transforms.Resample(_sr, self.sr)
            noise_wav = resampler(noise_wav)

            if ENABLE_PROFILING:
                step_times["resample"] = profile_step("resample", resample_start)

        if ENABLE_PROFILING:
            device_start = time.time()

        audio_t = wav.unsqueeze(0) if wav.ndim == 1 else wav  # shape (1,T) or (B,T)
        # Ensure noise tensor matches device
        noise_wav = noise_wav.to(audio_t.device)

        if ENABLE_PROFILING:
            step_times["device_transfer"] = profile_step(
                "device_transfer", device_start
            )

        if ENABLE_PROFILING:
            reshape_start = time.time()

        # Repeat / crop to match target audio length
        if noise_wav.shape[-1] < audio_len:
            reps = int(np.ceil(audio_len / noise_wav.shape[-1]))
            noise_wav = noise_wav.repeat(1, reps)
        if noise_wav.shape[-1] > audio_len:
            start = random.randint(0, noise_wav.shape[-1] - audio_len)
            noise_wav = noise_wav[:, start : start + audio_len]

        if ENABLE_PROFILING:
            step_times["reshape"] = profile_step("reshape", reshape_start)

        if ENABLE_PROFILING:
            mix_start = time.time()

        # --- Scale to random SNR -----------------------------------------
        signal_power = (audio_t**2).mean(dim=-1, keepdim=True)
        noise_power = (noise_wav**2).mean(dim=-1, keepdim=True)

        snr_db = random.uniform(*snr_db_range)
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-8))

        mixed = audio_t + scale * noise_wav

        if ENABLE_PROFILING:
            step_times["mixing"] = profile_step("mixing", mix_start)
            total_time = profile_step("total_noise_mixing", start_time)

            # Only print detailed report if total time exceeds threshold
            if total_time > 0.05:
                noise_file = os.path.basename(str(noise_path))
                logger.info(
                    f"[NOISE_PROFILE] {noise_file} | Total: {total_time:.4f}s | "
                    f"Steps: "
                    f"{' | '.join([f'{k}={v:.4f}s' for k, v in step_times.items()])}"
                )

                # Show cache stats occasionally but only for slow operations
                if (
                    hasattr(_cached_audio_info, "cache_info") and random.random() < 0.1
                ):  # 10% chance
                    cache_info = _cached_audio_info.cache_info()
                    logger.info(f"[CACHE_STATS] {cache_info}")

        # Print total time only if it exceeds threshold
        if ENABLE_PROFILING:
            if total_time > 0.05:
                logger.info(
                    f"[NOISE PROFILE] File: {os.path.basename(str(noise_path))} | "
                    f"Total mix time: {total_time:.4f}s"
                )

        return mixed.squeeze(0) if wav.ndim == 1 else mixed


class ItemPostprocessor:
    """
    A callable wrapper for AugmentationProcessor that handles NumPy <-> torch
    conversion.
    Use as a postprocessor in esp_data_temp datasets.

    Example usage:
        aug = AugmentationProcessor(...)
        postproc = ItemPostprocessor(aug)
        ds = get_dataset_dummy(..., postprocessors=[postproc])
    """

    def __init__(self, aug_processor: "AugmentationProcessor") -> None:
        """
        Initialize with an AugmentationProcessor.

        Parameters
        ----------
        aug_processor : AugmentationProcessor
            The augmentation processor to wrap
        """
        self.aug_processor = aug_processor

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply augmentations to an item, handling tensor conversion.

        Parameters
        ----------
        item : Dict[str, Any]
            The item dictionary with raw_wav and other fields

        Returns
        -------
        Dict[str, Any]
            The processed item with augmentations applied
        """
        import torch

        if not isinstance(item["raw_wav"], torch.Tensor):
            item["raw_wav"] = torch.from_numpy(item["raw_wav"])
        item = self.aug_processor.apply_augmentations(item)
        if isinstance(item["raw_wav"], torch.Tensor):
            item["raw_wav"] = item["raw_wav"].cpu().numpy()
        return item


def make_item_postprocessor(
    aug_processor: "AugmentationProcessor",
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a callable that wraps `aug_processor.apply_augmentations` for use as a
    postprocessor in esp_data_temp datasets. Handles NumPy <-> torch conversion.

    Example usage:
        aug = AugmentationProcessor(...)
        postproc = make_item_postprocessor(aug)
        ds = get_dataset_dummy(..., postprocessors=[postproc])

    Returns
    -------
    Callable[[Dict[str, Any]], Dict[str, Any]]
        A function that takes a sample dict, applies augmentations, and returns
        the processed dict.
    """
    return ItemPostprocessor(aug_processor)


# --------------------------------------------------------------------------- #
#  Helper utilities for faster *cached* noise mixing (used by AugmentationProcessor)
# --------------------------------------------------------------------------- #

# Global flag to enable/disable profiling
ENABLE_PROFILING = os.environ.get("PROFILE_NOISE_AUG", "1") == "1"

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure INFO logs are visible

# Global storage for profiling stats
_profiling_stats = defaultdict(list)


def get_profiling_summary() -> Dict[str, Dict[str, float]]:
    """Return a summary of profiling statistics.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary mapping step names to statistics (avg, max, min, count, total)
    """
    result = {}

    for key, values in _profiling_stats.items():
        if not values:
            continue

        # Calculate statistics
        avg = sum(values) / len(values)
        max_val = max(values)
        min_val = min(values)

        result[key] = {
            "avg": avg,
            "max": max_val,
            "min": min_val,
            "count": len(values),
            "total": sum(values),
        }

    return result


def print_profiling_summary() -> None:
    """Print a summary of profiling statistics."""
    summary = get_profiling_summary()

    logger.info("\n=== Noise Augmentation Profiling Summary ===")
    for step, stats in sorted(summary.items()):
        logger.info(
            f"{step:20s}: avg={stats['avg']:.4f}s | total={stats['total']:.2f}s | "
            f"count={stats['count']}"
        )
    logger.info("=" * 45 + "\n")


def profile_step(step_name: str, start_time: float) -> float:
    """Record timing for a profile step.

    Returns
    -------
    float
        The elapsed time in seconds
    """
    elapsed = time.time() - start_time
    if ENABLE_PROFILING:
        _profiling_stats[step_name].append(elapsed)
    return elapsed


def _list_noise_files(
    noise_dirs: Sequence[str],
    max_noise_samples: int = 10000,
) -> List[Path]:
    """Enumerate (once!) all candidate noise files across *noise_dirs*.

    The function is intentionally I/O heavy but is called **only once per
    process** during :pyclass:`AugmentationProcessor` initialisation.
    max_noise_samples applies to each noise directory.

    Returns
    -------
    List[Path]
        List of paths to audio files that can be used as noise

    Raises
    ------
    FileNotFoundError
        If any of the noise directories does not exist
    """

    noise_paths: List[Path] = []
    for dir_str in noise_dirs:
        dir_path: Path | "GSPath" = (
            GSPath(dir_str) if dir_str.startswith("gs://") else Path(dir_str)
        )
        if not dir_path.exists():
            raise FileNotFoundError(f"Noise directory not found: {dir_str}")

        for ext in (".wav", ".mp3", ".flac", ".ogg"):
            noise_files = list(dir_path.glob(f"*{ext}"))[:max_noise_samples]
            noise_paths.extend(noise_files)
    logger.info(f"Found {len(noise_paths)} noise files")
    return noise_paths


# Define a TypeVar for preserving the original callable's signature
CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def timing_decorator(func: CallableT) -> CallableT:
    """Decorator to time function calls.

    Returns
    -------
    CallableT
        A wrapped function that logs timing information
    """

    @wraps(func)
    # ruff: noqa: ANN001, ANN002, ANN003, ANN202
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"[TIMING] {func.__name__} took {elapsed:.4f}s")
        return result

    return cast(CallableT, wrapper)  # Cast back to the original type


def clear_info_cache() -> None:
    """Clear the torchaudio.info LRU cache for debugging."""
    if hasattr(_cached_audio_info, "cache_clear"):
        _cached_audio_info.cache_clear()
        logger.info("[PROFILE] Cleared _cached_audio_info cache")


def print_cache_stats() -> None:
    """Print current cache statistics."""
    if hasattr(_cached_audio_info, "cache_info"):
        info = _cached_audio_info.cache_info()
        logger.info(f"[PROFILE] Cache stats: {info}")


@lru_cache(maxsize=256)
def _cached_audio_info(path: str) -> torchaudio.AudioMetaData:
    """Lightweight wrapper around ``torchaudio.info`` with LRU caching.

    This is a performance critical function since it's called for every
    noise augmentation operation.

    Returns
    -------
    torchaudio.AudioMetaData
        Metadata for the audio file at the given path
    """
    start_time = time.time()
    try:
        info = torchaudio.info(path)
        _ = time.time() - start_time  # elapsed
        return info
    except Exception as e:
        logger.error(f"Error reading audio info from {path}: {e}")
        raise


def track_cache_performance(path: str) -> torchaudio.AudioMetaData:
    """Track cache performance for _cached_audio_info.

    Returns
    -------
    torchaudio.AudioMetaData
        The audio metadata object returned by _cached_audio_info
    """
    start_time = time.time()
    info = _cached_audio_info(path)
    elapsed = time.time() - start_time

    # Get cache stats after the call
    cache_info = _cached_audio_info.cache_info()
    logger.info(
        f"[CACHE INFO] Cache "
        f"{'hit' if cache_info.hits > cache_info.misses else 'MISS'} | "
        f"{path.split('/')[-1]} | time={elapsed:.4f}s | "
        f"hit_rate={(cache_info.hits / (cache_info.hits + cache_info.misses)):.2f} | "
        f"{cache_info.hits}/{cache_info.misses}"
    )

    return info
