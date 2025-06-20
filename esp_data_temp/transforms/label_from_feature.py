import logging
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from . import register_transform

logger = logging.Logger("esp_data")


class LabelFromFeatureConfig(BaseModel):
    type: Literal["label_from_feature"]
    feature: str
    label_map: dict[str, int] | None = None
    output_feature: str = "label"
    override: bool = False


class LabelFromFeature:
    def __init__(
        self,
        *,
        feature: str,
        label_map: dict[str, int] | None = None,
        output_feature: str = "label",
        override: bool = False,
    ) -> None:
        self.feature = feature
        self.label_map = label_map
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

        if self.label_map is None:
            uniques = sorted(df_clean[self.feature].unique())
            label_map = {lbl: idx for idx, lbl in enumerate(uniques)}
        else:
            label_map = self.label_map

        df_clean[self.output_feature] = df_clean[self.feature].map(label_map)


        df_clean["label_feature"] = df_clean[self.feature].apply(
            lambda x: [x] if not isinstance(x, list) else x
        )

        metadata = {
            "label_feature": self.feature,
            "label_map": label_map,
            "num_classes": len(label_map),
        }

        return df_clean, metadata


register_transform(LabelFromFeatureConfig, LabelFromFeature)
