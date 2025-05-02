"""
Data transformation utilities for filtering and subsampling datasets.
"""

import logging
from collections.abc import Callable
from dataclasses import field as dc_field
from functools import partial
from typing import Annotated, Any, Dict, List, Type, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Literal

logger = logging.Logger("esp_data")

class FilterConfig(BaseModel):
    type: Literal["filter"]
    mode: Literal["include", "exclude"] = "include"
    property: str
    values: List[str]


class SubsampleConfig(BaseModel):
    type: Literal["subsample"]
    property: str
    ratios: Dict[str, float] = dc_field(default_factory=dict)

    # TODO (milad) add a validator that does below:
    #         if not all(0 <= r <= 1 for r in config.ratios.values()):
    # raise ValueError("All ratios must be in [0, 1]")


class LabelFromFeature(BaseModel):
    type: Literal["label_from_feature"]
    feature: str
    num_classes: int | Literal["auto"] = "auto"
    output_feature: str = "label"
    override: bool = False


# TODO (this list in union should come from registry)
# TODO do we want a base model for transforms that forces "type"?
RegisteredTransforms = Annotated[
    Union[FilterConfig, SubsampleConfig, LabelFromFeature],
    Field(discriminator="type"),
]


# TODO (milad) I kind what all these transform functions to have a simple interface so
# maybe ban positional arguments once you've edited them all


# TODO (milad) name too similar too config class
# @register_transform(LabelFromFeature)
def create_labels(
    df: pd.DataFrame, cfg: LabelFromFeature
) -> tuple[pd.DataFrame, dict | None]:
    if cfg.output_feature in df and not cfg.override:
        assert False, "TODO (milad)"

    df_clean = df.dropna(subset=cfg.feature)
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


class Filter:
    """Filter data based on property values."""

    def __init__(self, config: FilterConfig):
        """
        Initialize the filter.

        Args:
            config: Filter configuration
        """
        self.config = config
        self.values = set(config.values)

    def __call__(
        self, data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Filter the data based on property values."""
        if isinstance(data, pd.DataFrame):
            return self._filter_dataframe(data), None
        elif isinstance(data, dict):
            return self._filter_dict(data), None
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a pandas DataFrame."""
        if self.config.mode == "include":
            return df[df[self.config.property].isin(self.values)]
        else:
            return df[~df[self.config.property].isin(self.values)]

    def _filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter a dictionary of data."""
        if self.config.mode == "include":
            return {
                k: v for k, v in data.items() if v[self.config.property] in self.values
            }
        else:
            return {
                k: v
                for k, v in data.items()
                if v[self.config.property] not in self.values
            }


class Subsample:
    """Subsample data based on property ratios."""

    def __init__(self, config: SubsampleConfig):
        if not all(0 <= r <= 1 for r in config.ratios.values()):
            raise ValueError("All ratios must be in [0, 1]")
        self.cfg = config

    def __call__(self, data: Union[pd.DataFrame, Dict[str, Any]]):
        if isinstance(data, pd.DataFrame):
            return self._subsample_dataframe(data), None
        if isinstance(data, dict):
            return self._subsample_dict(data), None
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _choose_keys(self, keys: List[Any], ratio: float) -> List[Any]:
        """Return a subsample of *keys* of size `ceil(len(keys)*ratio)`."""
        if ratio >= 1.0 or len(keys) == 0:
            return keys
        n = int(len(keys) * ratio)
        rng = np.random.default_rng(seed=42)
        return rng.choice(keys, size=n, replace=False).tolist()

    def _subsample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        prop = self.cfg.property
        ratios = self.cfg.ratios
        groups = []

        for val, ratio in ratios.items():
            if val == "other":
                continue
            idx = df.index[df[prop] == val].tolist()
            chosen = self._choose_keys(idx, ratio)
            groups.append(df.loc[chosen])

        if "other" in ratios:
            mask_other = ~df[prop].isin(ratios.keys() - {"other"})
            idx_other = df.index[mask_other].tolist()
            chosen_other = self._choose_keys(idx_other, ratios["other"])
            groups.append(df.loc[chosen_other])

        return pd.concat(groups, ignore_index=True)

    def _subsample_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prop = self.cfg.property
        ratios = self.cfg.ratios
        selected: Dict[str, Any] = {}

        for val, ratio in ratios.items():
            if val == "other":
                continue
            keys = [k for k, v in data.items() if v[prop] == val]
            for k in self._choose_keys(keys, ratio):
                selected[k] = data[k]

        if "other" in ratios:
            other_keys = [
                k for k, v in data.items() if v[prop] not in (ratios.keys() - {"other"})
            ]
            for k in self._choose_keys(other_keys, ratios["other"]):
                selected[k] = data[k]

        return selected


def build_transforms(transform_configs: list[RegisteredTransforms]) -> list[Callable]:
    """
    Build the transformation pipeline from **validated** configs.

    Parameters
    ----------
    transform_configs : list[RegisteredTransforms]
        The `transformations` field that comes straight out of a validated
        `DataConfig`.  No raw YAML dictionaries are accepted.

    Raises
    ------
    TypeError
        If the input is not a `FilterConfig` or `SubsampleConfig`.

    Returns
    -------
    list[Callable]
        Callable objects that can be applied in sequence.
    """
    transforms: list[Callable] = []

    # TODO (milad) replace with a Registry pattern?

    for cfg in transform_configs:
        if isinstance(cfg, FilterConfig):
            transforms.append(Filter(cfg))
        elif isinstance(cfg, SubsampleConfig):
            transforms.append(Subsample(cfg))
        elif isinstance(cfg, LabelFromFeature):
            transforms.append(partial(create_labels, cfg=cfg))  # TODO (milad)
        else:  # this should never happen if DataConfig was validated
            raise TypeError(
                "build_transforms() received an unexpected config type: "
                f"{type(cfg).__name__}"
            )

    return transforms
