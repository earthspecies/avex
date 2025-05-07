"""AnimalSpeak dataset"""

from io import StringIO
from typing import Literal

import pandas as pd
from registry import DatasetInfo, register_dataset, registry

from esp_data_temp.dataset import GSPath


@register_dataset
class AnimalSpeak:
    """AnimalSpeak dataset.

    Example:
    --------
    >>> from esp_data_temp.registered_datasets import AnimalSpeak
    >>> dataset = AnimalSpeak()
    >>> df = dataset.load("validation")
    >>> print(df.head())
    """

    info = DatasetInfo(
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
        if split not in self.info.split_paths:
            raise ValueError(
                f"""Invalid split: {split}.
                Expected one of {list(self.split_paths.keys())}"""
            )
        location = self.info.split_paths[split]
        # Read CSV content
        csv_text = GSPath(location).read_text(encoding="utf-8")
        df = pd.read_csv(StringIO(csv_text))
        df["gs_path"] = df["local_path"].apply(lambda x: "gs://" + x)

        return df

    # TODO
    # def save_to_disk / export

    # TODO (gagan)
    # def merge / concatenate (self, other_dataset: RegisteredDataset) -> None:


if __name__ == "__main__":
    # Example usage
    # print(registry.list())
    registry.print()
