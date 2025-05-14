"""
Data transformation utilities for filtering and subsampling datasets.
"""

import logging
from collections.abc import Callable
from functools import partial

import pandas as pd

from esp_data_temp.transforms import (
    Filter,
    FilterConfig,
    LabelFromFeature,
    RegisteredTransformConfigs,
    Subsample,
    SubsampleConfig,
    UniformSample,
    UniformSampleConfig,
    create_labels,
)

logger = logging.Logger("esp_data")


# TODO (milad) I kind what all these transform functions to have a simple interface so
# maybe ban positional arguments once you've edited them all


def build_transforms(
    transform_configs: list[RegisteredTransformConfigs],
) -> list[Callable[[pd.DataFrame], pd.DataFrame]]:
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
    transforms: list[Callable[[pd.DataFrame], pd.DataFrame]] = []

    # TODO (milad) replace with a Registry pattern?

    for cfg in transform_configs:
        if isinstance(cfg, FilterConfig):
            transforms.append(Filter(cfg))
        elif isinstance(cfg, SubsampleConfig):
            transforms.append(Subsample(cfg))
        elif isinstance(cfg, LabelFromFeature):
            transforms.append(partial(create_labels, cfg=cfg))  # TODO (milad)
        elif isinstance(cfg, UniformSampleConfig):
            transforms.append(UniformSample(cfg))
        else:  # this should never happen if DataConfig was validated
            raise TypeError(
                "build_transforms() received an unexpected config type: "
                f"{type(cfg).__name__}"
            )
    return transforms
