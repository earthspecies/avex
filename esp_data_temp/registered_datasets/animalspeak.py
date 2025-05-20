"""AnimalSpeak dataset"""

from io import StringIO
from typing import Any, Dict, Iterator, Literal, Sequence

import pandas as pd

from esp_data_temp.dataset import Dataset, DatasetInfo, GSPath, register_dataset


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
        super().__init__()  # Initialize the parent Dataset class
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
    def data(self) -> Sequence[Any]:
        """Get the dataset data.

        Returns
        -------
        Sequence[Any]
            The dataset data.
        """
        return self._data

    def _load(self, split: Literal["train", "validation"]) -> Sequence[Any]:
        """Load the given split of the dataset and return them.

        Parameters
        ----------
        splits : Literal["train", "validation"]
            Which split of the dataset to load. Can be "train" or "validation"
            for AnimalSpeak.

        Returns
        -------
        Sequence[Any]
            The corresponding split (as a pandas dataframe for now).

        Raises
        -------
        ValueError
            If the split is not valid.
        """

        if split not in self.info.split_paths:
            raise ValueError(
                f"""Invalid split: {split}.
                Expected one of {list(self.info.split_paths.keys())}"""
            )
        location = self.info.split_paths[split]
        # Read CSV content
        csv_text = GSPath(location).read_text(encoding="utf-8")
        self._data = pd.read_csv(StringIO(csv_text))
        self._data["gs_path"] = self._data["local_path"].apply(lambda x: "gs://" + x)
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

        # TODO: To adapt better to the dataset (reading audio, etc.)
        return row

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

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            A string representation of the dataset including its name, version,
            and basic statistics if data is loaded.
        """
        base_info = f"{self.info.name} (v{self.info.version})"
        if self._data is None:
            return f"{base_info} - No data loaded"

        return (
            f"{base_info}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Number of samples: {len(self)}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
