"""AnimalSpeak dataset"""

from io import StringIO
from typing import Literal

import pandas as pd
from registry import RegisteredDataset, register_dataset

from esp_data_temp.dataset import GSPath


@register_dataset
class AnimalSpeak(RegisteredDataset):
    """AnimalSpeak dataset.

    Example:
    --------
    >>> from esp_data_temp.registered_datasets import AnimalSpeak
    >>> dataset = AnimalSpeak()
    >>> df = dataset.load("validation")
    >>> print(df.head())
    """

    name: str = "animalspeak"
    owner: str = "david; marius; masato"
    split_paths: dict[str, str] = {
        "train": "gs://animalspeak2/splits/v1/animalspeak_train_v1.3.csv",
        "validation": "gs://animalspeak2/splits/v1/animalspeak_eval_v1.3.csv",
    }
    version: str = "0.1.0"
    description: str = "AnimalSpeak dataset"
    sources: list[str] = ["Xeno-canto", "iNaturalist", "Watkins"]
    license: str = "unknown"

    def load(self, split: Literal["train", "validation"]) -> pd.DataFrame:
        """Load the dataset from the specified location
        Arguments
        ---------
        split: str
            The split to load. Can be "train" or "validation".

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the dataset.

        Raises
        -------
        ValueError
            If the split is not valid.
        """
        if split not in self.split_paths:
            raise ValueError(
                f"""Invalid split: {split}.
                Expected one of {list(self.split_paths.keys())}"""
            )
        location = self.split_paths[split]
        # Read CSV content
        csv_text = GSPath(location).read_text(encoding="utf-8")
        df = pd.read_csv(StringIO(csv_text))
        df["gs_path"] = df["local_path"].apply(lambda x: "gs://" + x)

        return df

    # TODO
    # def save_to_disk / export

    # TODO (gagan)
    # def merge / concatenate (self, other_dataset: RegisteredDataset) -> None:
