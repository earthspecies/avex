from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from io import StringIO
import yaml

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

from representation_learning.data.data_utils import GSPath, get_dataset_from_name
from representation_learning.data.transformations import build_transforms
from representation_learning.configs import load_config
from representation_learning.data.audio_utils import pad_or_window  # type: ignore


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
        mask_tensor = torch.from_numpy(np.stack(masks))      # [B, T]  bool
        label_tensor = torch.tensor(labels, dtype=torch.long)

        out = {
            "raw_wav": audio_tensor,
            "padding_mask": mask_tensor,
            "label": label_tensor,
            "text_label": text_labels
        }
        return out

# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class AudioDataset(Dataset):
    """
    Reads metadata from a CSV, loads audio, and yields a sample dict.
    
    Expected columns in the CSV:
    * 'filepath'  : str – path to the audio file on disk or a gs:// path.
    * <label_col> : str – value used for the target (e.g. species name).
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_config: Any,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        preprocessor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_config = data_config
        self.preprocessor = preprocessor

        self.audio_path_col = "gs_path"  # modify if your CSV uses a different name
        self.label_col = data_config.label_column

        # Build a label → index mapping for numeric targets
        unique_labels = sorted(self.metadata[self.label_col].unique())
        self.label2idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(unique_labels)}

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        path_str: str = row[self.audio_path_col]

        # Use GSPath for gs:// paths if available, otherwise use the local Path.
        if path_str.startswith("gs://"):
            if GSPath is None:
                raise ImportError("cloudpathlib is required to handle gs:// paths.")
            audio_path = GSPath(path_str)
        else:
            audio_path = Path(path_str)

        # Open the audio file. Using the .open('rb') method works for both local and GSPath objects.
        with audio_path.open("rb") as f:
            audio, sr = sf.read(f)
        if audio.ndim == 2:  # stereo → mono
            audio = audio.mean(axis=1)

        return {
            "raw_wav": audio.astype(np.float32),
            "text_label": row[self.label_col],
            "label": self.label2idx[row[self.label_col]],
            "path": str(audio_path),
        }


def get_dataset_dummy(
    data_config: Any,
    preprocessor: Optional[Callable] = None,
    validation: bool = False
) -> AudioDataset:
    """
    Dataset entry point that supports both local and GS paths, with transformations.
    
    1. Loads metadata CSV (path specified in `data_config.dataset_source`).
    2. Applies any filtering / subsampling specified in `data_config.transformations`.
    3. Returns an `AudioDataset` instance.
    """
    
    # Check if the dataset CSV path is a gs:// path
    df = get_dataset_from_name(data_config.dataset_name, validation)

    # Apply transformations if specified
    if hasattr(data_config, 'transformations') and data_config.transformations:
        transforms = build_transforms(data_config.transformations)
        for transform in transforms:
            df = transform(df)

    return AudioDataset(
        metadata_df=df,
        data_config=data_config,
        preprocessor=preprocessor,
    )


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
    
    data_config = load_config(cfg.dataset_config, config_type = "data")
    
    # Create dataset using the updated get_dataset_dummy
    ds_train = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,  # Add any audio preprocessing here if needed
        validation=True #TEMP: for testing speed
    )
    ds_eval = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,  # Add any audio preprocessing here if needed
        validation=True
    )
    
    # Create collater
    collate_fn = Collater(
        audio_max_length_seconds=cfg.model_spec.audio_config.target_length,
        sr=cfg.model_spec.audio_config.sample_rate,
        window_selection=cfg.model_spec.audio_config.window_selection
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