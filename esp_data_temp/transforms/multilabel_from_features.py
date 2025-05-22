import logging
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from . import register_transform

logger = logging.Logger("esp_data")


class MultiLabelFromFeaturesConfig(BaseModel):
    type: Literal["labels_from_features"]
    features: str | list[str]
    num_classes: int | Literal["auto"] = "auto"
    output_feature: str = "label"
    override: bool = False


class MultiLabelFromFeatures:
    """
    A transform that generates multi-label targets from one or more feature columns.

    This class goes through one or more specified columns and generates a mapping of
    unique values to integer IDs. It then uses this mapping to generate a new column
    where each row contains a list of integer label IDs corresponding to the unique
    values found in the specified feature columns. It is useful for preparing data for
    multi-label classification tasks, where each sample may be associated with multiple
    labels.

    Notes
    -----
    If element values are themselves lists, the transform will explode them first before
    constructing the mapping dictionary and converting the values.

    Parameters
    ----------
    features : list[str]
        The names of the columns in the DataFrame to use as sources for the labels. Each
        column can contain a single value or a list of values per row.
    num_classes : int or "auto", default="auto"
        The number of unique classes. If set to "auto", the number of classes is
        inferred from the data.
    output_feature : str, default="label"
        The name of the output column to store the generated label lists.
    override : bool, default=False
        If False and the output_feature already exists in the dataset, an error is
        raised. If True, the output_feature will be overwritten.

    Methods
    -------
    from_config(cfg: MultiLabelFromFeaturesConfig) -> MultiLabelFromFeatures
        Instantiates the transform from a configuration object.
    __call__(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]
        Applies the transform to the DataFrame, returning the modified DataFrame and
        metadata about the label mapping.
    """

    def __init__(
        self,
        *,
        features: list[str],
        num_classes: int | Literal["auto"] = "auto",
        output_feature: str = "label",
        override: bool = False,
    ) -> None:
        self.features = features
        self.num_classes = num_classes
        self.override = override
        self.output_feature = output_feature

    @classmethod
    def from_config(cls, cfg: MultiLabelFromFeaturesConfig) -> "MultiLabelFromFeatures":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if self.output_feature in df and not self.override:
            raise AssertionError("TODO (milad)")

        uniques = set()
        for f in self.features:
            # explode() turns empty lists into NaNs hence the dropna()
            uniques |= set(df[f].explode().dropna().unique())

        label_map = {lbl: idx for idx, lbl in enumerate(sorted(uniques))}

        def _row_to_ids(row: pd.Series) -> list | None:
            row_labels = []
            for f in self.features:
                if isinstance(row[f], list):
                    v = row[f]
                elif pd.isna(row[f]):
                    continue
                else:
                    v = [row[f]]
                row_labels.extend(map(lambda x: label_map[x], v))

            if row_labels:
                return sorted(row_labels)
            else:
                return None

        df[self.output_feature] = df[self.features].apply(_row_to_ids, axis="columns")

        df_clean = df.dropna(subset=self.output_feature)

        if len(df_clean) != len(df):
            logger.warning(
                f"Dropped {len(df) - len(df_clean)} rows with {self.output_feature}=NaN"
            )

        metadata = {
            "label_feature": self.features,
            "label_map": label_map,
            "num_classes": len(uniques)
            if self.num_classes == "auto"
            else self.num_classes,
        }

        return df_clean, metadata


register_transform(MultiLabelFromFeaturesConfig, MultiLabelFromFeatures)
