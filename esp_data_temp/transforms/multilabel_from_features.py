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

        def _row_to_ids(row: pd.Series) -> list:
            row_labels = []
            for f in self.features:
                if not isinstance(row[f], list):
                    v = [row[f]]
                else:
                    v = row[f]
                row_labels.extend(map(lambda x: label_map[x], v))
            return sorted(row_labels)

        df[self.output_feature] = df[self.features].apply(_row_to_ids, axis="columns")

        metadata = {
            "label_feature": self.features,
            "label_map": label_map,
            "num_classes": len(uniques)
            if self.num_classes == "auto"
            else self.num_classes,
        }

        return df, metadata


register_transform(MultiLabelFromFeaturesConfig, MultiLabelFromFeatures)
