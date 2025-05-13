import os
from collections.abc import Callable
from functools import lru_cache
from io import StringIO
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Iterator, List, Optional, Self, Type

import cloudpathlib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from google.cloud.storage.client import Client

from .config import DatasetConfig
from .transformations import build_transforms

ANIMALSPEAK_PATH = "gs://animalspeak2/splits/v1/animalspeak_train_v1.3_cluster.csv"
ANIMALSPEAK_PATH_EVAL = "gs://animalspeak2/splits/v1/animalspeak_eval_v1.3_cluster.csv"
BATS_PATH = "gs://foundation-model-data/audio/egyptian_fruit_bats/annotations.train.csv"
BATS_PATH_VALID = (
    "gs://foundation-model-data/audio/egyptian_fruit_bats/annotations.valid.csv"
)
BATS_PATH_TEST = (
    "gs://foundation-model-data/audio/egyptian_fruit_bats/annotations.test.csv"
)


@lru_cache(maxsize=1)
def _get_client() -> cloudpathlib.GSClient:
    return cloudpathlib.GSClient(storage_client=Client(), file_cache_mode="close_file")


class GSPath(cloudpathlib.GSPath):
    """
    A wrapper for the cloudpathlib GSPath that provides a default client.
    This avoids issues when the GOOGLE_APPLICATION_CREDENTIALS variable is not set.
    """

    def __init__(
        self,
        client_path: str | Self | cloudpathlib.AnyPath,
        client: cloudpathlib.GSClient = _get_client(),
    ) -> None:
        super().__init__(client_path, client=client)


class AudioDataset:
    """
    Reads metadata from a CSV, loads audio, and yields a sample dict.

    Expected columns in the CSV:
    * 'filepath'  : str - path to the audio file on disk or a gs:// path.
    * <label_col> : str - value used for the target (e.g. species name).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_config: DatasetConfig,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        preprocessor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        metadata: dict | None = None,
        postprocessors: Optional[
            List[Callable[[Dict[str, Any]], Dict[str, Any]]]
        ] = None,
    ) -> None:
        super().__init__()

        # TODO (milad) transform arg here?

        self.df = df.reset_index(drop=True)
        self.data_config = data_config
        self.preprocessor = preprocessor

        self.audio_path_col = "gs_path"  # modify if your CSV uses a different name

        self.metadata = metadata

        self.postprocessors = postprocessors or []

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        # TODO
        pass

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator:
        # TODO (milad) do this properly
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        # TODO
        return "TODO"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        path_str: str = row[self.audio_path_col]

        # Use GSPath for gs:// paths if available, otherwise use the local Path.
        if path_str.startswith("gs://"):
            if GSPath is None:
                raise ImportError("cloudpathlib is required to handle gs:// paths.")
            audio_path = GSPath(path_str)
        else:
            audio_path = Path(path_str)

        # Open the audio file. Using the .open('rb') method works for both local and
        # GSPath objects.
        with audio_path.open("rb") as f:
            audio, sr = sf.read(f)
        if audio.ndim == 2:  # stereo → mono
            audio = audio.mean(axis=1)

        if "sample_rate" in self.data_config and sr != self.data_config.sample_rate:
            resampler = librosa.resampler.Resampler(
                orig_sr=sr,
                target_sr=self.data_config.sample_rate,
                res_type="kaiser_fast",
            )
            audio = resampler(audio)
            sr = self.data_config.sample_rate

        item = {
            "raw_wav": audio.astype(np.float32),
            "text_label": row["label"],  # TODO (milad) we assume supervisor, fix
            "label": row.label,
            "path": str(audio_path),
        }

        for proc in self.postprocessors:
            item = proc(item)

        return item


def _get_dataset_from_name(
    name: str,
    split: str = "train",
) -> pd.DataFrame:
    name = name.lower().strip()

    if name == "animalspeak":
        if split == "test":
            return None
        anaimspeak_path = (
            ANIMALSPEAK_PATH_EVAL if split == "valid" else ANIMALSPEAK_PATH
        )
        if ANIMALSPEAK_PATH.startswith("gs://"):
            csv_path = GSPath(anaimspeak_path)
        else:
            csv_path = Path(anaimspeak_path)

        # Read CSV content
        csv_text = csv_path.read_text(encoding="utf-8")
        df = pd.read_csv(StringIO(csv_text))
        df["gs_path"] = df["local_path"].apply(
            lambda x: "gs://" + x
        )  # AnimalSpeak missing gs path
        return df
    elif name == "bats":
        csv_file = (
            BATS_PATH_TEST
            if split == "test"
            else BATS_PATH_VALID
            if split == "valid"
            else BATS_PATH
        )
        # TODO: don't use os.path!
        base_path = os.path.dirname(csv_file).split("egyptian_fruit_bats")[0]
        if csv_file.startswith("gs://"):
            csv_path = GSPath(csv_file)
        else:
            csv_path = Path(csv_file)

        # Read CSV content
        csv_text = csv_path.read_text(encoding="utf-8")
        df = pd.read_csv(StringIO(csv_text))
        df["gs_path"] = df["path"].apply(
            lambda x: base_path
            + "egyptian_fruit_bats"
            + x.split("egyptian_fruit_bats")[1]
        )  # bats missing gs path
        return df
    else:
        raise NotImplementedError("Dataset not supported")


def get_dataset_dummy(
    data_config: DatasetConfig,
    split: str,
    preprocessor: Optional[Callable] = None,
    postprocessors: Optional[
        List[Callable[[Dict[str, Any]], Dict[str, Any]]]
    ] = None,
) -> AudioDataset:
    """
    Dataset entry point that supports both local and GS paths, with transformations.

    1. Loads datasets
    2. Applies any filtering / subsampling specified in `data_config.transformations`.
    3. Returns an `AudioDataset` instance.

    Parameters
    ----------
    data_config : DataConfig
        Configuration for the dataset
    preprocessor : Optional[Callable]
        Optional preprocessor function
    split : bool
        Whether to split the dataset

    Returns
    -------
    AudioDataset
        An instance of the dataset with the specified transformations applied.
    """

    # Check if the dataset CSV path is a gs:// path
    df = _get_dataset_from_name(data_config.dataset_name, split)

    metadata = {}

    if data_config.transformations:
        transforms = build_transforms(data_config.transformations)
        for transform in transforms:
            df, md = transform(df)

            # TODO (milad): hacky but let's think about it
            # TODO (test if keys already exist and shout?)
            if md:
                metadata.update(md)

    return AudioDataset(
        df=df,
        data_config=data_config,
        preprocessor=preprocessor,
        metadata=metadata,
        postprocessors=postprocessors,
    )
