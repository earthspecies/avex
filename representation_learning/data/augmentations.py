"""
Audio augmentation utilities.

This module provides functions for audio data augmentation, including:
- add_noise: Add background noise to audio at specified signal-to-noise ratio (SNR)
- mixup: Mixup augmentation for audio samples
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torchaudio

from representation_learning.configs import RunConfig


def add_noise(
    audio: Union[np.ndarray, torch.Tensor],
    noise_dir: Union[str, List[str]],
    snr_db_range: Tuple[float, float] = (-5.0, 20.0),
    sample_rate: int = 16000,
    max_noise_samples: int = 1000,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Add background noise to the input audio at a random SNR within the specified range.

    Parameters
    ----------
    audio : Union[np.ndarray, torch.Tensor]
        Input audio signal, shape (samples,) or (batch, samples)
    noise_dir : Union[str, List[str]]
        Directory or list of directories containing noise audio files
    snr_db_range : Tuple[float, float], optional
        Range of signal-to-noise ratio in dB, by default (-5.0, 20.0)
    sample_rate : int, optional
        Sample rate of the audio signal, by default 16000
    max_noise_samples : int, optional
        Maximum number of noise samples to load, by default 1000

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Noisy audio signal with same type and shape as input

    Raises
    ------
    FileNotFoundError
        If a noise directory does not exist
    ValueError
        If no audio files are found in the noise directories
    RuntimeError
        If loading a noise file fails (e.g., GCS access issue or corrupted file)

    Notes
    -----
    This implementation follows the approach described in the NatureLM-audio paper,
    which adds background noise at varying SNR levels to improve model robustness.
    """
    # Check if input is tensor or numpy array
    is_tensor = isinstance(audio, torch.Tensor)
    is_1d = len(audio.shape) == 1

    # Convert to tensor if numpy array
    if not is_tensor:
        audio_tensor = torch.from_numpy(audio)
    else:
        audio_tensor = audio

    # Ensure audio is 2D: (batch_size, samples)
    if is_1d:
        audio_tensor = audio_tensor.unsqueeze(0)

    # Get audio length and prepare noise paths
    audio_length = audio_tensor.shape[-1]
    noise_dirs = [noise_dir] if isinstance(noise_dir, str) else noise_dir
    noise_paths = []

    from esp_data_temp.dataset import GSPath

    # Collect all noise file paths
    for dir_path in noise_dirs:
        # Use GSPath for gs:// paths if available, otherwise use local Path
        if dir_path.startswith("gs://"):
            path = GSPath(dir_path)
        else:
            path = Path(dir_path)

        if not path.exists():
            raise FileNotFoundError(f"Noise directory not found: {dir_path}")

        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
            noise_paths.extend(list(path.glob(f"*{ext}")))

        # Limit the number of noise samples to avoid memory issues
        if len(noise_paths) > max_noise_samples:
            noise_paths = random.sample(noise_paths, max_noise_samples)
            break

    if not noise_paths:
        raise ValueError(f"No audio files found in the noise directories: {noise_dirs}")

    # Randomly select a noise file
    noise_path = random.choice(noise_paths)

    # Load noise audio
    # torchaudio.load should handle CloudPath objects if fsspec/gcsfs are installed
    try:
        noise, noise_sr = torchaudio.load(noise_path)  # type: ignore
        noise = noise.to(audio_tensor.device)
    except Exception as e:
        # Provide more context if loading fails, potentially due to GCS issues
        raise RuntimeError(f"Failed to load noise file {noise_path}: {e}") from e

    # Convert to mono if stereo
    if noise.shape[0] > 1:
        noise = torch.mean(noise, dim=0, keepdim=True)

    # Resample if necessary
    if noise_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=noise_sr, new_freq=sample_rate
        )
        noise = resampler(noise)

    # If noise is shorter than the target audio, loop it
    if noise.shape[1] < audio_length:
        num_repeats = int(np.ceil(audio_length / noise.shape[1]))
        noise = noise.repeat(1, num_repeats)

    # If noise is longer than the target audio, select a random segment
    if noise.shape[1] > audio_length:
        start = random.randint(0, noise.shape[1] - audio_length)
        noise = noise[:, start : start + audio_length]

    # Calculate signal and noise power
    signal_power = torch.mean(audio_tensor**2, dim=1, keepdim=True)
    noise_power = torch.mean(noise**2, dim=1, keepdim=True)

    # Get random SNR within specified range
    snr_db = random.uniform(snr_db_range[0], snr_db_range[1])

    # Calculate the scaling factor for the noise
    snr = 10 ** (snr_db / 10)
    # Add small epsilon to prevent division by zero
    scale = torch.sqrt(signal_power / (noise_power * snr + 1e-8))

    # Scale noise and add to signal
    scaled_noise = scale * noise
    noisy_audio = audio_tensor + scaled_noise

    # Return the same type and dimensionality as input
    if is_1d:
        noisy_audio = noisy_audio.squeeze(0)

    if not is_tensor:
        return noisy_audio.numpy()
    else:
        return noisy_audio


def mixup(
    audio1: Union[np.ndarray, torch.Tensor],
    audio2: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.2,
) -> Tuple[Union[np.ndarray, torch.Tensor], float]:
    """
    TODO: Mixup not yet tested!
    Apply mixup augmentation to two audio samples.

    Parameters
    ----------
    audio1 : Union[np.ndarray, torch.Tensor]
        First audio signal
    audio2 : Union[np.ndarray, torch.Tensor]
        Second audio signal, must have same shape as audio1
    alpha : float, optional
        Alpha parameter for beta distribution, by default 0.2

    Returns
    -------
    Tuple[Union[np.ndarray, torch.Tensor], float]
        Tuple of (mixed audio, lambda)
        lambda is the mixing coefficient from Beta(alpha, alpha)

    Notes
    -----
    Implementation of mixup as described in "mixup: Beyond Empirical Risk Minimization"
    (Zhang et al., 2017)
    """
    # Check if inputs are tensors or numpy arrays
    is_tensor = isinstance(audio1, torch.Tensor)

    # Convert to tensors if numpy arrays
    if not is_tensor:
        audio1_tensor = torch.from_numpy(audio1)
        audio2_tensor = torch.from_numpy(audio2)
    else:
        audio1_tensor = audio1
        audio2_tensor = audio2

    # Sample lambda from beta distribution
    lam = np.random.beta(alpha, alpha)

    # Mix the audios
    mixed_audio = lam * audio1_tensor + (1 - lam) * audio2_tensor

    # Return the same type as input
    if not is_tensor:
        return mixed_audio.numpy(), lam
    else:
        return mixed_audio, lam


class AugmentationProcessor:
    """Applies audio augmentations based on configuration."""

    def __init__(self, cfg: RunConfig, device: str = "cpu") -> None:
        """
        Initialize the augmentation processor.

        Parameters
        ----------
        cfg : RunConfig
            Configuration containing augmentation settings
        device : str, optional
            Device to perform augmentations on, by default "cpu"
        """
        self.augmentations = cfg.augmentations
        self.device = device
        self.sr = cfg.sr
        from representation_learning.configs import (  # local import to avoid circular
            MixupAugment,
            NoiseAugment,
        )

        self.noise_augs = [
            aug for aug in self.augmentations if isinstance(aug, NoiseAugment)
        ]
        self.mixup_augs = [
            aug for aug in self.augmentations if isinstance(aug, MixupAugment)
        ]

    def apply_augmentations(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configured augmentations to a batch.

        Parameters
        ----------
        batch : Dict[str, Any]
            Batch dictionary containing 'raw_wav' tensor (B, T)

        Returns
        -------
        Dict[str, Any]
            Augmented batch with same structure
        """
        import random

        import torch
        import torch.nn.functional as F

        from .augmentations import add_noise, mixup

        augmented_batch = batch.copy()

        # Get the audio from the batch
        audio = batch["raw_wav"].to(self.device)

        # Apply noise augmentation
        for noise_aug in self.noise_augs:
            if random.random() < noise_aug.augmentation_prob:
                # Apply noise augmentation to each sample in the batch
                for i in range(audio.shape[0]):
                    if (
                        random.random() < noise_aug.augmentation_prob
                    ):  # Per-sample probability
                        audio[i] = add_noise(
                            audio[i],
                            noise_dir=noise_aug.noise_dirs,
                            snr_db_range=noise_aug.snr_db_range,
                            sample_rate=self.sr,
                        )

        # Apply mixup augmentation
        for mixup_aug in self.mixup_augs:
            if random.random() < mixup_aug.augmentation_prob:
                # Create pairs for mixup by shuffling the batch
                indices = torch.randperm(audio.shape[0], device=self.device)
                shuffled_audio = audio[indices]
                shuffled_labels = batch["label"][indices]

                # Apply mixup to the entire batch
                mixed_audio, lam = mixup(audio, shuffled_audio, alpha=mixup_aug.alpha)

                # Update audio and create mixed labels
                audio = mixed_audio

                # For supervised learning, we need to update the labels
                if "label" in batch and not isinstance(batch["label"], list):
                    # One-hot encode labels for mixup
                    n_classes = batch["label"].max().item() + 1
                    y_a = F.one_hot(batch["label"], num_classes=n_classes).float()
                    y_b = F.one_hot(shuffled_labels, num_classes=n_classes).float()
                    mixed_labels = lam * y_a + (1 - lam) * y_b
                    augmented_batch["mixed_labels"] = mixed_labels

        # Update the batch with augmented audio
        augmented_batch["raw_wav"] = audio

        return augmented_batch
