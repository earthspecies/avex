from __future__ import annotations

import multiprocessing
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from esp_data_temp.dataset import get_dataset_dummy
from representation_learning.configs import RunConfig, load_config
from representation_learning.data.audio_utils import (
    pad_or_window,  # type: ignore
)


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
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders from configuration.

    Parameters
    ----------
    cfg : RunConfig
        Run configuration containing dataset and training parameters
    device : str
        Device to use for data loading

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    if device != "cpu":
        multiprocessing.set_start_method("spawn", force=True)

    # Load dataset configuration
    data_config = load_config(cfg.dataset_config, config_type="data")

    # Create augmentation processor if augmentations are defined
    # train_aug_processor = None
    # if cfg.augmentations:
    #     aug_device = "cpu"  # Augmentations in dataloader should ideally be CPU-bound
    #     train_aug_processor = AugmentationProcessor(
    #         cfg.augmentations, cfg.sr, aug_device
    #     )

    # Create dataset using the updated get_dataset_dummy
    ds_train = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,
        validation=cfg.debug_mode,
    )
    ds_eval = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,
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

    return train_dl, val_dl
