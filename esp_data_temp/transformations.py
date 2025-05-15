"""
Data transformation utilities for filtering and subsampling datasets.
"""

import logging
from collections.abc import Callable

import pandas as pd

from esp_data_temp.transforms import (
    Filter,
    FilterConfig,
    LabelFromFeature,
    LabelFromFeatureConfig,
    RegisteredTransformConfigs,
    Subsample,
    SubsampleConfig,
    UniformSample,
    UniformSampleConfig,
)

logger = logging.Logger("esp_data")


# TODO (milad) I kind what all these transform functions to have a simple interface so
# maybe ban positional arguments once you've edited them all


def transform_from_config(
    cfg: RegisteredTransformConfigs,
) -> Callable[[pd.DataFrame], tuple[pd.DataFrame, dict]]:
    """
    Build the transformation pipeline from a list of Pydantic-validated configs.

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
    # TODO (milad) replace with a Registry pattern?

    if isinstance(cfg, FilterConfig):
        return Filter.from_config(cfg)
    elif isinstance(cfg, SubsampleConfig):
        return Subsample(cfg)
    elif isinstance(cfg, LabelFromFeatureConfig):
        return LabelFromFeature.from_config(cfg)
    elif isinstance(cfg, UniformSampleConfig):
        return UniformSample(cfg)
    else:  # this should never happen if DataConfig was validated
        raise TypeError(
            "build_transforms() received an unexpected config type: "
            f"{type(cfg).__name__}"
        )
