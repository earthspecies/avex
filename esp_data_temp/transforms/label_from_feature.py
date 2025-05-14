import logging
from typing import Literal

import pandas as pd

from ._base import TransformModel

logger = logging.Logger("esp_data")


class LabelFromFeature(TransformModel):
    type: Literal["label_from_feature"]
    feature: str
    num_classes: int | Literal["auto"] = "auto"
    output_feature: str = "label"
    override: bool = False


# TODO (milad) name too similar too config class
# @register_transform(LabelFromFeature)
def create_labels(
    df: pd.DataFrame, cfg: LabelFromFeature
) -> tuple[pd.DataFrame, dict | None]:
    if cfg.output_feature in df and not cfg.override:
        raise AssertionError("TODO (milad)")

    # TODO (milad) the .copy() is probably making this slow but without it I get this
    # warning: https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#copy-on-write-chained-assignment
    # Find a better way
    df_clean = df.dropna(subset=[cfg.feature]).copy()
    if len(df_clean) != len(df):
        logger.warn(f"Dropped {len(df) - len(df_clean)} rows with {cfg.feature}=NaN")

    uniques = sorted(df_clean[cfg.feature].unique())
    label_mapping = {lbl: idx for idx, lbl in enumerate(uniques)}
    df_clean[cfg.output_feature] = df_clean[cfg.feature].map(label_mapping)

    # TODO (milad): hacky. Just here to make things run
    # We should think about how transforms can add/modify dataset-wide properties
    metadata = {
        "label_map": label_mapping,
        "num_classes": len(uniques) if cfg.num_classes == "auto" else cfg.num_classes,
    }

    return df_clean, metadata
