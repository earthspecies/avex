from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Self

import cloudpathlib
import numpy as np
import pandas as pd
import soundfile as sf
from google.cloud.storage.client import Client

from .config import DataConfig
from .transformations import (
    DataTransform,
    Filter,
    FilterConfig,
    Subsample,
    SubsampleConfig,
    TransformCfg,
)

ANIMALSPEAK_PATH = "gs://animalspeak2/splits/v1/animalspeak_train_v1.3.csv"
ANIMALSPEAK_PATH_EVAL = "gs://animalspeak2/splits/v1/animalspeak_eval_v1.3.csv"


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
        client_path: str | Self | "CloudPath",
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
        metadata_df: pd.DataFrame,
        data_config: DataConfig,
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

    # TODO (milad) we mostly care about iteration so define __iter__

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

        # Open the audio file. Using the .open('rb') method works for both local and
        # GSPath objects.
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


def _build_transforms(transform_configs: List[TransformCfg]) -> List[DataTransform]:
    """
    Build the transformation pipeline from **validated** configs.

    Parameters
    ----------
    transform_configs : list[FilterConfig | SubsampleConfig]
        The `transformations` field that comes straight out of a validated
        `DataConfig`.  No raw YAML dictionaries are accepted.

    Raises
    ------
    TypeError
        If the input is not a `FilterConfig` or `SubsampleConfig`.

    Returns
    -------
    list[DataTransform]
        Callable objects that can be applied in sequence.
    """
    transforms: List[DataTransform] = []

    for cfg in transform_configs:
        if isinstance(cfg, FilterConfig):
            transforms.append(Filter(cfg))
        elif isinstance(cfg, SubsampleConfig):
            transforms.append(Subsample(cfg))
        else:  # this should never happen if DataConfig was validated
            raise TypeError(
                "build_transforms() received an unexpected config type: "
                f"{type(cfg).__name__}"
            )

    return transforms


def _get_dataset_from_name(
    name: str,
    validation: bool = False,
) -> pd.DataFrame:
    name = name.lower().strip()

    if name == "animalspeak":
        anaimspeak_path = ANIMALSPEAK_PATH_EVAL if validation else ANIMALSPEAK_PATH
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
    else:
        raise NotImplementedError("Only AnimalSpeak dataset supported")


def get_dataset_dummy(
    data_config: DataConfig,
    preprocessor: Optional[Callable] = None,
    validation: bool = False,
) -> AudioDataset:
    """
    Dataset entry point that supports both local and GS paths, with transformations.

    1. Loads metadata CSV (path specified in `data_config.dataset_source`).
    2. Applies any filtering / subsampling specified in `data_config.transformations`.
    3. Returns an `AudioDataset` instance.

    Returns
    -------
    AudioDataset
        An instance of the dataset with the specified transformations applied.
    """

    # Check if the dataset CSV path is a gs:// path
    df = _get_dataset_from_name(data_config.dataset_name, validation)

    # Apply transformations if specified
    if hasattr(data_config, "transformations") and data_config.transformations:
        transforms = _build_transforms(data_config.transformations)
        for transform in transforms:
            df = transform(df)

    return AudioDataset(
        metadata_df=df,
        data_config=data_config,
        preprocessor=preprocessor,
    )
