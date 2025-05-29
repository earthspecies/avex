"""BarkleyCanyon dataset
References
----------
K. S. Kanes,
“Recycling data: An annotated marine acoustic data set that
is publicly available for use  in classifier development
and marine mammal research,”
The Journal of the Acoustical Society of America, vol. 148,
"""

import pathlib
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import librosa
import numpy as np
import pandas as pd
from esp_data.io import GSPath

from esp_data_temp.config import DatasetConfig
from esp_data_temp.datasets.base import (
    Dataset,
    DatasetInfo,
    register_dataset,
)

from .dataset_utils import read_audio_clip


@register_dataset
class BarkleyCanyon(Dataset):
    """BarkleyCanyon dataset.

    Example:
    --------
    """

    info = DatasetInfo(
        name="barkley_canyon",
        owner="gagan",
        split_paths={
            "train": "gs://esp-ml-datasets/barkley_canyon/v0.1.0/raw/barkley_canyon_annotations_non_null_gbif_v2.csv",
        },
        version="0.1.0",
        description="BarkleyCanyon dataset",
        sources=["Barkley Canyon"],
        license="unknown",
    )

    def __init__(
        self,
        split: str = "train",
        output_take_and_give: dict[str, str] = None,
        sample_rate: int = 16000,
        audio_path_col: str = "gs_path",
        postprocessors: Optional[
            List[Callable[[Dict[str, Any]], Dict[str, Any]]]
        ] = None,
    ) -> None:
        """Initialize the AnimalSpeak dataset.

        Parameters
        ----------
        split : str
            The split to load. One of info.split_paths keys.
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        sample_rate : int
            The sample rate to which audio files should be resampled.
        audio_path_col : str
            The name of the column in the DataFrame that contains the audio file paths.
        postprocessors : Optional[List[Callable[[Dict[str, Any]], Dict[str, Any]]]]
            A list of post-processing functions to apply to each sample after loading.
        """
        super().__init__(output_take_and_give)  # Initialize the parent Dataset class
        self._data: pd.DataFrame = None
        self._load(split)  # Load the dataset (fills self._data)
        self.sample_rate = sample_rate
        self.audio_path_col = audio_path_col
        self.postprocessors = postprocessors or []

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    def _load(self, split: str) -> None:
        """Load the given split of the dataset and return them.

        Parameters
        ----------
        split : str
            Which split of the dataset to load. Must be one of info.split_paths keys.

        Raises
        ------
        ValueError
            If the split is not valid.
        """
        if split not in self.info.split_paths:
            raise ValueError(
                f"Invalid split: {split}. "
                f"Expected one of {list(self.info.split_paths.keys())}"
            )

        location = self.info.split_paths[split]
        # Read CSV content
        csv_text = GSPath(location).read_text(encoding="utf-8")
        self._data = pd.read_csv(StringIO(csv_text))

        # Add with a gs:// prefix to the local_path column
        location_dir = GSPath(location).parent
        self._data["gs_path"] = self._data["local_path"].apply(
            lambda x: location_dir / x
        )

    @classmethod
    def from_config(cls, cfg: DatasetConfig) -> "BarkleyCanyon":
        """Create a CSVDataset instance from a configuration dictionary.

        Parameters
        ----------
        cfg : DatasetConfig
            Configuration dictionary containing dataset parameters.

        Returns
        -------
        CSVAudioDataset
            An instance of the CSVDataset class.

        Raises
        -------
        ValueError
            If the configuration is missing required fields or contains invalid values.
        """
        cfg = cfg.model_dump(exclude=("dataset_name", "transformations"))

        split = cfg.get("split", None)
        if not split or split not in cls.info.split_paths:
            raise ValueError(
                f"Invalid split '{split}'."
                f"Available splits: {', '.join(cls.info.split_paths.keys())}"
            )
        if "audio_path_col" not in cfg:
            raise ValueError(
                "Configuration must include 'audio_path_col' to specify the column"
                "in the underlying dataframe containing audio file paths."
            )
        if "sample_rate" not in cfg:
            raise ValueError(
                "Configuration must include 'sample_rate' to "
                "specify the target sample rate for audio."
            )

        output_take_and_give = cfg.get("output_take_and_give", None)
        return cls(split=split, output_take_and_give=output_take_and_give)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call load() first.")
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.
        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the audio data, text label, label, and path.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if idx < 0 or idx >= len(self._data):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self._data)}."
            )

        row = self._data.iloc[idx].to_dict()
        path_str: str = row[self.audio_path_col]

        # Use GSPath for gs:// paths if available, otherwise use the local Path.
        # TODO (gagan / milad) Replace with esp_data.io
        if isinstance(path_str, GSPath) or isinstance(path_str, pathlib.Path):
            audio_path = path_str
        elif str(path_str).startswith("gs://"):
            audio_path = GSPath(path_str)
        else:
            audio_path = Path(path_str)

        audio, sr = read_audio_clip(
            audio_path,
            start_time=row["start_times(sec)"],
            end_time=row["end_times(sec)"],
            to_mono=True,
            mono_method="average",
        )
        audio = audio.astype(np.float32)

        target_sr = self.sample_rate
        if target_sr is not None and sr != target_sr:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=target_sr,
                scale=True,
                res_type="kaiser_best",
            )
            sr = target_sr

        row["audio"] = audio

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        # FIXME: This looks like transforms ? Remove ?
        for proc in self.postprocessors:
            item = proc(item)

        return item

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        Dict[str, Any]
            Each sample in the dataset.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call load() first.")

        for idx in range(len(self)):
            yield self[idx]

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            A string representation of the dataset including its name, version,
            and basic statistics if data is loaded.
        """
        base_info = f"{self.info.name} (v{self.info.version})"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
