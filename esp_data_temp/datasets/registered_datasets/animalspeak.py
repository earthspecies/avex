"""AnimalSpeak dataset"""

from io import StringIO
from typing import Any, Dict, Iterator, Literal, Sequence

import pandas as pd

from esp_data_temp.datasets.base import (
    Dataset,
    DatasetInfo,
    GSPath,
    register_dataset,
)


@register_dataset
class AnimalSpeak(Dataset):
    """AnimalSpeak dataset.

    Example:
    --------
    >>> from esp_data_temp.datasets import AnimalSpeak
    >>> dataset = AnimalSpeak(split="validation",
                    output_take_and_give={"species_common": "comm"})
    >>> print(dataset.info)
    """

    info = DatasetInfo(
        name="animalspeak",
        owner="david; marius; masato",
        split_paths={
            "train": "gs://animalspeak2/splits/v1/animalspeak_train_v1.3_cluster.csv",
            "validation": "gs://animalspeak2/splits/v1/animalspeak_eval_v1.3_cluster.csv",
        },
        version="0.1.0",
        description="AnimalSpeak dataset",
        sources=["Xeno-canto", "iNaturalist", "Watkins"],
        license="unknown",
    )

    def __init__(
        self, split: str = "train", output_take_and_give: dict[str, str] = None
    ) -> None:
        """Initialize the AnimalSpeak dataset.

        Parameters
        ----------
        split : str
            The split to load. Can be "train" or "validation".
        output_take_and_give : dict[str, str]
            A dictionary mapping the original column names to the new column names.
            It acts as a filter as well.
        """
        super().__init__(output_take_and_give)  # Initialize the parent Dataset class
        self._data: pd.DataFrame = None
        self._load(split)  # Load the dataset (fills self._data)

    def _load(self, split: Literal["train", "validation"]) -> Sequence[Any]:
        """Load the given split of the dataset and return them.

        Parameters
        ----------
        splits : Literal["train", "validation"]
            Which split of the dataset to load. Can be "train" or "validation"
            for AnimalSpeak.

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

        if idx < 0 or idx >= len(self._data):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self._data)}."
            )
        row = self._data.iloc[idx].to_dict()
        if self.output_take_and_give:
            mapped_row = {}
            for key, value in self.output_take_and_give.items():
                mapped_row[value] = row[key]
            return mapped_row
        else:
            return row

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
