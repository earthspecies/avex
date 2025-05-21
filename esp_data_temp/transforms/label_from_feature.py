import logging
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from . import register_transform

logger = logging.Logger("esp_data")


class LabelFromFeatureConfig(BaseModel):
    type: Literal["label_from_feature"]
    feature: str
    num_classes: int | Literal["auto"] = "auto"
    output_feature: str = "label"
    override: bool = False


class LabelFromFeature:
    def __init__(
        self,
        *,
        feature: str,
        num_classes: int | Literal["auto"] = "auto",
        output_feature: str = "label",
        override: bool = False,
    ) -> None:
        self.feature = feature
        self.num_classes = num_classes
        self.override = override
        self.output_feature = output_feature

    @classmethod
    def from_config(cls, cfg: LabelFromFeatureConfig) -> "LabelFromFeature":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if self.output_feature in df and not self.override:
            raise AssertionError("TODO (milad)")

        # TODO (milad) the .copy() is probably making this slow but without it I get the
        # warning below. Maybe find a better way?
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#copy-on-write-chained-assignment
        df_clean = df.dropna(subset=[self.feature]).copy()
        if len(df_clean) != len(df):
            logger.warning(
                f"Dropped {len(df) - len(df_clean)} rows with {self.feature}=NaN"
            )

        uniques = sorted(df_clean[self.feature].unique())
        label_mapping = {lbl: idx for idx, lbl in enumerate(uniques)}
        df_clean[self.output_feature] = df_clean[self.feature].map(label_mapping)

        # TODO (milad): hacky. Just here to make things run
        # We should think about how transforms can add/modify dataset-wide properties
        metadata = {
            "label_feature": self.feature,
            "label_map": label_mapping,
            "num_classes": len(uniques)
            if self.num_classes == "auto"
            else self.num_classes,
        }

        return df_clean, metadata


register_transform(LabelFromFeatureConfig, LabelFromFeature)
