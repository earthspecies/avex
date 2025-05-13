from __future__ import annotations

import os
import random
from collections.abc import Callable, Iterator
from functools import lru_cache
from io import StringIO
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, Self, Type

import cloudpathlib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf  # Third-party I/O
from google.cloud.storage.client import Client

from .config import DatasetConfig
from .transformations import build_transforms

# --------------------------------------------------------------------------- #
# Dataset CSV locations
# --------------------------------------------------------------------------- #
ANIMALSPEAK_PATH = "gs://animalspeak2/splits/v1/animalspeak_train_v1.3_cluster.csv"
ANIMALSPEAK_PATH_EVAL = "gs://animalspeak2/splits/v1/animalspeak_eval_v1.3_cluster.csv"

BATS_PATH = "gs://foundation-model-data/audio/egyptian_fruit_bats/annotations.train.csv"
BATS_PATH_VALID = (
    "gs://foundation-model-data/audio/egyptian_fruit_bats/annotations.valid.csv"
)
BATS_PATH_TEST = (
    "gs://foundation-model-data/audio/egyptian_fruit_bats/annotations.test.csv"
)

# --------------------------------------------------------------------------- #
# Google-Cloud helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _get_client() -> cloudpathlib.GSClient:  # pragma: no cover
    return cloudpathlib.GSClient(storage_client=Client(), file_cache_mode="close_file")


_default_client = _get_client()


class GSPath(cloudpathlib.GSPath):
    """Wrapper that injects a default GSClient so callers don't need env vars."""

    def __init__(
        self,
        client_path: str | Self | cloudpathlib.AnyPath,
        client: cloudpathlib.GSClient = _default_client,
    ) -> None:
        super().__init__(client_path, client=client)


# --------------------------------------------------------------------------- #
# Core dataset
# --------------------------------------------------------------------------- #
class AudioDataset:
    """
    Simple iterable dataset that:

    1. Reads filepaths & labels from a CSV.
    2. Extracts a ≤ 60 s window (random if the file is longer).
    3. Optionally resamples and applies caller-supplied post-processors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_config: DatasetConfig,
        *,
        preprocessor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        postprocessors: Optional[
            list[Callable[[dict[str, Any]], dict[str, Any]]]
        ] = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        # Store dataframe & configs early (needed before validation below)
        self.df = df.reset_index(drop=True)
        self.data_config = data_config
        self.preprocessor = preprocessor
        self.postprocessors = postprocessors or []
        self.extra_metadata = metadata or {}

        # ------------------------------------------------------------------ #
        # Label handling
        # ------------------------------------------------------------------ #
        self.label_col: str = getattr(data_config, "label_column", "label")
        if self.label_col not in self.df.columns:
            raise ValueError(
                f"Label column '{self.label_col}' not found in metadata."
            )

        unique_labels = sorted(self.df[self.label_col].unique())
        self.label2idx: dict[str, int] = {
            lbl: i for i, lbl in enumerate(unique_labels)
        }

        # Column holding the actual audio path
        self.audio_path_col = "gs_path"

    # --------------------- Python protocol helpers -------------------------- #
    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n={len(self)}, "
            f"labels={len(self.label2idx)}, "
            f"config={self.data_config.dataset_name})"
        )

    # Optional context-manager hooks (nothing special to clean up)
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        return None

    # -------------------------- Data access --------------------------------- #
    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        path_str: str = row[self.audio_path_col]

        # Resolve gs:// → local caching or direct file path
        audio_path = (
            GSPath(path_str) if path_str.startswith("gs://") else Path(path_str)
        )

        # ---------- 1) Lightweight header read ---------- #
        with audio_path.open("rb") as f:
            info = sf.info(f)
            sr = info.samplerate
            n_frames = info.frames

            window_frames = min(sr * 60, n_frames)
            start_fr = (
                random.randint(0, n_frames - window_frames)
                if n_frames > window_frames
                else 0
            )
            stop_fr = start_fr + window_frames

            try:
                f.seek(0)  # rewind after sf.info
                audio, _ = sf.read(
                    f,
                    start=start_fr,
                    stop=stop_fr,
                    dtype="float32",
                )
            except Exception as e:
                audio, _ = sf.read(
                    f,
                    dtype="float32",
                )

        # Convert stereo → mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Optional pre-resample
        target_sr: int | None = getattr(self.data_config, "sample_rate", None)
        if target_sr and sr != target_sr:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast"
            )
            sr = target_sr

        # Optional caller-supplied preprocessing
        if self.preprocessor is not None:
            audio = self.preprocessor(audio, sr)

        # Label mapping (single-label classification assumed here)
        raw_lbl = row[self.label_col]
        int_lbl = self.label2idx[raw_lbl]

        item: dict[str, Any] = {
            "raw_wav": audio.astype(np.float32),
            "text_label": raw_lbl,
            "label": int_lbl,
            "path": str(audio_path),
            "sample_rate": sr,
            **self.extra_metadata,
        }

        for proc in self.postprocessors:
            item = proc(item)

        return item


# --------------------------------------------------------------------------- #
# Dataset builders
# --------------------------------------------------------------------------- #
def _get_dataset_from_name(name: str, *, split: str = "train") -> pd.DataFrame:
    """Return a *metadata* DataFrame for a known dataset/split.

    Parameters
    ----------
    name : str
        Identifier of the dataset (e.g. ``"animalspeak"``).
    split : str, optional
        One of ``"train"``, ``"valid"``, or ``"test"`` specifying the split
        to load.

    Returns
    -------
    pandas.DataFrame
        A dataframe with at minimum a ``gs_path`` column pointing to the audio
        files on disk or in cloud storage.

    Raises
    ------
    ValueError
        If *name* / *split* combination is invalid (e.g. requesting a test
        split that does not exist).
    NotImplementedError
        If the dataset *name* is unknown.
    """
    name = name.lower().strip()

    if name == "animalspeak":
        if split == "test":
            raise ValueError("AnimalSpeak does not provide a test split yet.")
        csv_file = ANIMALSPEAK_PATH if split == "train" else ANIMALSPEAK_PATH_EVAL

        csv_path: Path | GSPath = (
            GSPath(csv_file) if csv_file.startswith("gs://") else Path(csv_file)
        )
        df = pd.read_csv(StringIO(csv_path.read_text(encoding="utf-8")))

        # Convert relative `local_path` → absolute path on local FS (cluster)
        df["gs_path"] = df["local_path"].apply(
            lambda x: (
                "/home/milad_earthspecies_org/data-migration/marius-highmem/mnt/"
                "foundation-model-data/audio_16k/" + x
            )
        )
        return df

    elif name == "bats":
        csv_file = (
            BATS_PATH_TEST
            if split == "test"
            else BATS_PATH_VALID
            if split == "valid"
            else BATS_PATH
        )
        base = os.path.dirname(csv_file).split("egyptian_fruit_bats")[0]
        csv_path: Path | GSPath = (
            GSPath(csv_file) if csv_file.startswith("gs://") else Path(csv_file)
        )

        df = pd.read_csv(StringIO(csv_path.read_text(encoding="utf-8")))
        df["gs_path"] = df["path"].apply(
            lambda p: base + "egyptian_fruit_bats" + p.split("egyptian_fruit_bats")[1]
        )
        return df

    else:
        raise NotImplementedError(f"Dataset '{name}' not supported.")


def get_dataset_dummy(
    data_config: DatasetConfig,
    *,
    split: str = "train",
    preprocessor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    postprocessors: Optional[list[Callable[[dict[str, Any]], dict[str, Any]]]] = None,
) -> AudioDataset:
    """Return an :class:`AudioDataset` for quick experiments.

    This helper:

    1. Loads datasets
    2. Applies any filtering / subsampling specified in `data_config.transformations`.
    3. Returns an `AudioDataset` instance.

    Returns
    -------
    AudioDataset
        The constructed dataset ready for use in a PyTorch ``DataLoader``.
    """

    df = _get_dataset_from_name(data_config.dataset_name, split=split)

    # Apply declarative transforms (filtering / subsampling)
    metadata: dict[str, Any] = {}
    if getattr(data_config, "transformations", None):
        for transform in build_transforms(data_config.transformations):
            df, md = transform(df)
            if md:
                metadata.update(md)

    return AudioDataset(
        df=df,
        data_config=data_config,
        preprocessor=preprocessor,
        postprocessors=postprocessors,
        metadata=metadata,
    )
