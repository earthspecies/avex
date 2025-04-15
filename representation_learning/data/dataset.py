
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from representation_learning.data.audio_utils import pad_or_window  # type: ignore


logger = logging.getLogger(__name__)


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
                target_len=self.audio_max_length_seconds,
                window_selection=self.window_selection,
                sr=self.sr
            )
            audios.append(wav)
            masks.append(pad_mask)
            labels.append(item["label"])
            if self.keep_text:
                text_labels.append(item["text_label"])

        audio_tensor = torch.from_numpy(np.stack(audios))  # [B, T]  float32
        mask_tensor = torch.from_numpy(np.stack(masks))    # [B, T]  bool
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

    Expected columns in the CSV
    ---------------------------
    * 'filepath'  : str – path to the audio file on disk
    * <label_col> : str – value used for the target (e.g. species name)
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
        self.transform = transform
        self.preprocessor = preprocessor

        self.audio_path_col = "filepath"  # modify if your CSV uses a different name
        self.label_col = data_config.label_column

        # Build a label → index mapping for numeric targets
        unique_labels = sorted(self.metadata[self.label_col].unique())
        self.label2idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(unique_labels)}

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]

        audio_path = Path(row[self.audio_path_col])
        label_str: str = row[self.label_col]
        label_idx: int = self.label2idx[label_str]

        # --- Load audio -------------------------------------------------------
        audio, sr = sf.read(audio_path)
        if audio.ndim == 2:  # stereo → mono
            audio = audio.mean(axis=1)

        # --- Optional augmentation / transform --------------------------------
        if self.transform is not None:
            audio = self.transform(audio)

        return {
            "raw_wav": audio.astype(np.float32),
            "text_label": label_str,
            "label": label_idx,
            "path": str(audio_path),
        }


def get_dataset_dummy(
    data_config: Any,
    transform: Optional[Callable] = None,
    preprocessor: Optional[Callable] = None,
) -> AudioDataset:
    """
    TODO: ---This is the dummy dataset entry point----
    It will be replaced with the real interface.
    -----
    1. Load metadata CSV (path in `data_config.dataset_source`).
    2. Apply any filtering / subsampling outside this function (or add here).
    3. Return an `AudioDataset` instance.
    -----
    """
    csv_path = Path(data_config.dataset_name)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # TODO: apply transformations described in
    #       data_config.transformations here

    return AudioDataset(
        metadata_df=df,
        data_config=data_config,
        transform=transform,
        preprocessor=preprocessor,
    )
