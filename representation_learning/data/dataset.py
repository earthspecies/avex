from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from esp_data_temp.dataset import get_dataset_dummy
from representation_learning.configs import load_config
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
    ) -> None:
        self.audio_max_length_seconds = audio_max_length_seconds
        self.window_selection = window_selection
        self.keep_text = keep_text
        self.preprocessor = preprocessor
        self.sr = sr

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        audios, masks, labels = [], [], []
        text_labels: List[str] = []

        for item in batch:
            wav, pad_mask = pad_or_window(
                item["raw_wav"],
                target_len=self.audio_max_length_seconds * self.sr,
                window_selection=self.window_selection,
            )
            audios.append(wav)
            masks.append(pad_mask)
            labels.append(item["label"])
            if self.keep_text:
                text_labels.append(item["text_label"])

        audio_tensor = torch.from_numpy(np.stack(audios))  # [B, T]  float32
        mask_tensor = torch.from_numpy(np.stack(masks))  # [B, T]  bool
        label_tensor = torch.tensor(labels, dtype=torch.long)

        out = {
            "raw_wav": audio_tensor,
            "padding_mask": mask_tensor,
            "label": label_tensor,
            "text_label": text_labels,
        }
        return out


def build_dataloaders(cfg, device="cpu"):
    """
    Build training and validation dataloaders from configuration.

    Args:
        cfg: Run configuration
        device: Device to use for data loading

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Load dataset configuration

    data_config = load_config(cfg.dataset_config, config_type="data")

    # Create dataset using the updated get_dataset_dummy
    ds_train = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,  # Add any audio preprocessing here if needed
        validation=True,  # TEMP: for testing speed
    )
    ds_eval = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,  # Add any audio preprocessing here if needed
        validation=True,
    )

    # Create collater
    collate_fn = Collater(
        audio_max_length_seconds=cfg.model_spec.audio_config.target_length,
        sr=cfg.model_spec.audio_config.sample_rate,
        window_selection=cfg.model_spec.audio_config.window_selection,
    )

    # Create dataloaders
    train_dl = DataLoader(
        ds_train,
        batch_size=cfg.training_params.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    val_dl = DataLoader(
        ds_eval,
        batch_size=cfg.training_params.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    return train_dl, val_dl
