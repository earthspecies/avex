"""esp-data transform: drop rows where any of the specified columns are
missing or empty."""

from __future__ import annotations

from typing import List, Literal

import pandas as pd
from esp_data.transforms import register_transform
from pydantic import BaseModel


class RequireFeaturesConfig(BaseModel):
    type: Literal["require_features"]
    features: List[str]


class RequireFeatures:
    """Drop rows that have NA/empty values in any of the required columns."""

    def __init__(self, *, features: List[str]) -> None:
        self.features = features

    @classmethod
    def from_config(cls, cfg: RequireFeaturesConfig) -> "RequireFeatures":
        return cls(features=cfg.features)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: ANN001
        mask = pd.Series(True, index=df.index)
        for col in self.features:
            if col not in df.columns:
                # Treat completely missing columns as all-NA so we drop the row
                mask &= False
                continue
            col_vals = df[col]
            na_mask = col_vals.isna()
            if col_vals.dtype == "object":
                empty_str = col_vals.astype(str).str.strip() == ""
                na_mask |= empty_str
            mask &= ~na_mask
        filtered = df[mask].reset_index(drop=True)
        metadata = {
            "rows_dropped": int((~mask).sum()),
            "required_features": self.features,
        }
        return filtered, metadata


register_transform(RequireFeaturesConfig, RequireFeatures)
