"""AnimalSpeak dataset"""

from io import StringIO
from typing import Any, Dict, Iterator, Literal, Optional

import pandas as pd
import soundfile as sf
import librosa
import numpy as np
from esp_data_temp.registered_datasets import DatasetInfo, register_dataset, registry
from esp_data_temp.dataset import Dataset, GSPath


@register_dataset
class AnimalSpeak(Dataset):
    """AnimalSpeak dataset.

    Example:
    --------
    >>> from esp_data_temp.registered_datasets import AnimalSpeak
    >>> dataset = AnimalSpeak()
    >>> df = dataset.load("validation")
    >>> print(df.head())
    """

    def __init__(self):
        """Initialize the AnimalSpeak dataset."""
        self._data = None
        self._info = DatasetInfo(
            name="animalspeak",
            owner="david; marius; masato",
            split_paths={
                "train": "gs://animalspeak2/splits/v1/animalspeak_train_v1.3.csv",
                "validation": "gs://animalspeak2/splits/v1/animalspeak_eval_v1.3.csv",
            },
            version="0.1.0",
            description="AnimalSpeak dataset",
            sources=["Xeno-canto", "iNaturalist", "Watkins"],
            license="unknown",
        )

        self.load(["train", "validation"])

    @property
    def info(self) -> DatasetInfo:
        """Get the dataset information.
        
        Returns
        -------
        DatasetInfo
            Object containing dataset metadata like name, version, paths, etc.
        """
        return self._info

    @property
    def data(self) -> pd.DataFrame:
        """Get the current dataframe.
        
        Returns
        -------
        pd.DataFrame
            The current loaded split as a dataframe.
        
        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call load() first.")
        return self._data

    def load(
        self, 
        split: List[str] = ["train", "validation"]
    ) -> pd.DataFrame:
        """Load the given split(s) of the dataset and return them.
        
        Parameters
        ----------
        split : List[str]
            Which split(s) of the dataset to load. Can be "train" and/or "validation"
            for AnimalSpeak.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary mapping split names to their corresponding pandas DataFrames.

        Raises
        -------
        ValueError
            If the split is not valid.
        """
        self._data = {}
        for split in splits:
            if split not in self.info.split_paths:
                raise ValueError(
                    f"""Invalid split: {split}.
                    Expected one of {list(self.info.split_paths.keys())}"""
                )
            location = self.info.split_paths[split]
            # Read CSV content
            csv_text = GSPath(location).read_text(encoding="utf-8")
            self._data[split] = pd.read_csv(StringIO(csv_text))
            self._data[split]["gs_path"] = self._data[split]["local_path"].apply(lambda x: "gs://" + x)
        return self._data

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the sample data with keys:
            - 'raw_wav': The audio waveform as a numpy array
            - 'text_label': The text label for the sample
            - 'label': The numeric label for the sample
            - 'path': The path to the audio file
            
        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        IndexError
            If the index is out of bounds.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call load() first.")
        
        row = self._data.iloc[idx]
        path_str = row["gs_path"]

        # Use GSPath for gs:// paths
        audio_path = GSPath(path_str)

        # Load and process audio
        with audio_path.open("rb") as f:
            audio, sr = sf.read(f)
        if audio.ndim == 2:  # stereo → mono
            audio = audio.mean(axis=1)

        return {
            "raw_wav": audio.astype(np.float32),
            "text_label": row["label"],
            "label": row["label"],
            "path": str(audio_path),
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset.
        
        Returns
        -------
        Iterator[Dict[str, Any]]
            Iterator over samples in the dataset.
            
        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call load() first.")
        
        for idx in range(len(self)):
            yield self[idx]


if __name__ == "__main__":
    # Example usage
    # print(registry.list())
    registry.print()
