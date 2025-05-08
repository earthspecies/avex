"""
Data transformation utilities for filtering and subsampling datasets.
"""

from abc import ABC, abstractmethod
from dataclasses import field as dc_field
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing_extensions import Literal


class FilterConfig(BaseModel):
    property: str
    values: List[str]
    operation: Literal["include", "exclude"] = "include"


class SubsampleConfig(BaseModel):
    property: str
    operation: Literal["subsample"] = "subsample"
    ratios: Dict[str, float] = dc_field(default_factory=dict)


TransformCfg = Union[FilterConfig, SubsampleConfig]


class DataTransform(ABC):
    """Base class for data transformations."""

    @abstractmethod
    def __call__(
        self, data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Apply the transformation to the data.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.
        """
        pass


class Filter(DataTransform):
    """Filter data based on property values."""

    def __init__(self, config: FilterConfig) -> None:
        """
        Initialize the filter.

        Args:
            config: Filter configuration

        Raises:
            ValueError: If the operation is not 'include' or 'exclude'.
        """
        self.config = config
        self.values = set(config.values)

        if config.operation not in ["include", "exclude"]:
            raise ValueError(
                f"Operation must be 'include' or 'exclude', got {config.operation}"
            )

    def __call__(
        self, data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Filter the data based on property values.

        Args:
            data: The data to filter (DataFrame or dict).

        Returns:
            The filtered data (same type as input).

        Raises:
            TypeError: If the data type is not supported.
        """
        if isinstance(data, pd.DataFrame):
            return self._filter_dataframe(data)
        elif isinstance(data, dict):
            return self._filter_dict(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a pandas DataFrame.

        Args:
            df: The DataFrame to filter.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if self.config.operation == "include":
            return df[df[self.config.property].isin(self.values)]
        else:
            return df[~df[self.config.property].isin(self.values)]

    def _filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter a dictionary of data.

        Args:
            data: The dictionary to filter.

        Returns:
            Dict[str, Any]: The filtered dictionary.
        """
        if self.config.operation == "include":
            return {
                k: v for k, v in data.items() if v[self.config.property] in self.values
            }
        else:
            return {
                k: v
                for k, v in data.items()
                if v[self.config.property] not in self.values
            }


class Subsample(DataTransform):
    """Subsample data based on property ratios."""

    def __init__(self, config: SubsampleConfig) -> None:
        """
        Initialize the subsample transform.

        Args:
            config: Subsample configuration

        Raises:
            ValueError: If the operation is not 'subsample' or ratios are not in [0, 1].
        """
        if config.operation != "subsample":
            raise ValueError("SubsampleConfig.operation must be 'subsample'")
        if not all(0 <= r <= 1 for r in config.ratios.values()):
            raise ValueError("All ratios must be in [0, 1]")
        self.cfg = config

    def __call__(
        self, data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Apply the subsample transformation.

        Args:
            data: The data to subsample (DataFrame or dict).

        Returns:
            The subsampled data (same type as input).

        Raises:
            TypeError: If the data type is not supported.
        """
        if isinstance(data, pd.DataFrame):
            return self._subsample_dataframe(data)
        if isinstance(data, dict):
            return self._subsample_dict(data)
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _choose_keys(self, keys: List[Any], ratio: float) -> List[Any]:
        """Return a subsample of *keys* of size `ceil(len(keys)*ratio)`.

        Args:
            keys: List of keys to subsample from.
            ratio: Ratio of keys to select.

        Returns:
            List[Any]: The selected keys.
        """
        if ratio >= 1.0 or len(keys) == 0:
            return keys
        n = int(len(keys) * ratio)
        rng = np.random.default_rng(seed=42)
        return rng.choice(keys, size=n, replace=False).tolist()

    def _subsample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subsample a pandas DataFrame.

        Args:
            df: The DataFrame to subsample.

        Returns:
            pd.DataFrame: The subsampled DataFrame.
        """
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
        """Subsample a dictionary of data.

        Args:
            data: The dictionary to subsample.

        Returns:
            Dict[str, Any]: The subsampled dictionary.
        """
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
