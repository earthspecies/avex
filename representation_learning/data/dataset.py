from __future__ import annotations

import multiprocessing
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

from esp_data_temp.dataset import get_dataset_dummy
from representation_learning.configs import (
    Augment,
    MixupAugment,
    NoiseAugment,
    RunConfig,
    load_config,
)
from representation_learning.data.audio_utils import (
    pad_or_window,  # type: ignore
)
from representation_learning.data.augmentations import add_noise, mixup
from representation_learning.training.distributed import is_slurm_available


# --------------------------------------------------------------------------- #
#  Augmentation Processor
# --------------------------------------------------------------------------- #
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
                    y_b = F.one_hot(
                        shuffled_labels, num_classes=n_classes
                    ).float()
                    mixed_labels = lam * y_a + (1 - lam) * y_b
                    augmented_batch["mixed_labels"] = mixed_labels

        # Update the batch with augmented audio
        augmented_batch["raw_wav"] = audio

        return augmented_batch


# --------------------------------------------------------------------------- #
#  Collater
# --------------------------------------------------------------------------- #
class Collater:
    """
    Combines samples into a batch, ensuring every audio clip has the same
    length (`audio_max_length`) by truncating or zero‑padding as needed.
    """

    def __init__(
        self,
        audio_max_length_seconds: int,
        sr: int,
        window_selection: str = "random",
        keep_text: bool = False,
        preprocessor: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.audio_max_length_seconds = audio_max_length_seconds
        self.window_selection = window_selection
        self.keep_text = keep_text
        self.preprocessor = preprocessor
        self.sr = sr
        self.device = device

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        audios, masks, labels, text_labels = [], [], [], []

        for item in batch:
            wav = torch.as_tensor(item["raw_wav"])  # (T,)
            wav, pad_mask = pad_or_window(
                wav, self.audio_max_length_seconds * self.sr, self.window_selection
            )
            audios.append(wav)
            masks.append(pad_mask)
            labels.append(item["label"])
            if self.keep_text:
                text_labels.append(item["text_label"])

        # Keep tensors on CPU for pinning
        audio_tensor = torch.stack(audios)  # [B, T] float32
        mask_tensor = torch.stack(masks)  # [B, T] bool
        label_tensor = torch.tensor(labels, dtype=torch.long)

        return {
            "raw_wav": audio_tensor,
            "padding_mask": mask_tensor,
            "label": label_tensor,
            "text_label": text_labels,
        }


def build_dataloaders(
    cfg: RunConfig, device: str = "cpu"
) -> Tuple[DataLoader, DataLoader, Optional[AugmentationProcessor]]:
    """Build training and validation dataloaders from configuration.

    Parameters
    ----------
    cfg : RunConfig
        Run configuration containing dataset and training parameters
    device : str
        Device to use for data loading

    Returns
    -------
    Tuple[DataLoader, DataLoader, Optional[AugmentationProcessor]]
        Tuple of (train_dataloader, val_dataloader, augmentation_processor)
    """
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    if device != "cpu":
        multiprocessing.set_start_method("spawn", force=True)

    # Load dataset configuration
    data_config = load_config(cfg.dataset_config, config_type="data")

    # Create dataset using the updated get_dataset_dummy
    ds_train = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,  # Add any audio preprocessing here if needed
        validation=cfg.debug_mode,  # Use validation set in debug mode
    )
    ds_eval = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,  # Add any audio preprocessing here if needed
        validation=True,
    )

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(ds_train)
        val_sampler = DistributedSampler(ds_eval, shuffle=False)

    # Create collater
    collate_fn = Collater(
        audio_max_length_seconds=cfg.model_spec.audio_config.target_length_seconds,
        sr=cfg.model_spec.audio_config.sample_rate,
        window_selection=cfg.model_spec.audio_config.window_selection,
        keep_text=(cfg.label_type == "text"),  # Keep text labels for CLIP training
        device=device,
    )

    # Create dataloaders
    train_dl = DataLoader(
        ds_train,
        batch_size=cfg.training_params.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    val_dl = DataLoader(
        ds_eval,
        batch_size=cfg.training_params.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    # Create augmentation processor if augmentations are defined
    aug_processor = None
    if cfg.augmentations:
        aug_processor = AugmentationProcessor(cfg, device)

    return train_dl, val_dl, aug_processor
