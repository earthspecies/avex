from __future__ import annotations

import multiprocessing
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from esp_data_temp.dataset import get_dataset_dummy
from representation_learning.configs import RunConfig, DataConfig, load_config
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
    cfg: RunConfig, 
    data_config: DataConfig = None, 
    device: str = "cpu",
    subset_percentage: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders from configuration.

    Parameters
    ----------
    cfg : RunConfig
        Run configuration containing dataset and training parameters
    data_config : DataConfig
        Data configuration containing dataset details
    device : str
        Device to use for data loading
    subset_percentage : float
        Percentage of data to use (0.0 to 1.0)

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    if device != "cpu":
        multiprocessing.set_start_method("spawn", force=True)
        
    if data_config is None:
        dataset_config = cfg.dataset_config
        # Load dataset configuration
        data_config = load_config(dataset_config, config_type="data")

    # Create dataset using the updated get_dataset_dummy
    ds_train = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,  # Add any audio preprocessing here if needed
        split="train",
        subset_percentage=subset_percentage,
    )
    ds_val = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,  # Add any audio preprocessing here if needed
        split="valid",
        subset_percentage=subset_percentage,
    )

    ds_test = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,  # Add any audio preprocessing here if needed
        split="test",
        subset_percentage=subset_percentage,
    )

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
        ds_val,
        batch_size=cfg.training_params.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    if ds_test is not None:
        test_dl = DataLoader(
            ds_test,
            batch_size=cfg.training_params.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device != "cpu"),
        )
    else:
        test_dl = None

    return train_dl, val_dl, test_dl
