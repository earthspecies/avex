import logging
from typing import Any, Dict, Literal, Union

import pandas as pd

from ._base import TransformModel

logger = logging.Logger("esp_data")


class FilterConfig(TransformModel):
    type: Literal["filter"]
    mode: Literal["include", "exclude"] = "include"
    property: str
    values: list[str]


class Filter:
    """Filter data based on property values."""

    def __init__(
        self,
        *,
        property: str,
        values: list[str],
        mode: Literal["include", "exclude"] = "include",
    ) -> None:
        """
        Initialize the filter.
        """

        self.mode = mode
        self.property = property
        self.values = values

    @classmethod
    def from_config(cls, cfg: FilterConfig) -> "Filter":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(
        self, data: pd.DataFrame | dict[str, Any]
    ) -> tuple[pd.DataFrame | dict[str, Any], dict]:
        """Filter the data based on property values.

        Args:
            data: The data to filter (DataFrame or dict).

        Returns:
            The filtered data (same type as input).

        Raises:
            TypeError: If the data type is not supported.
        """
        if isinstance(data, pd.DataFrame):
            return self._filter_dataframe(data), {}
        elif isinstance(data, dict):
            return self._filter_dict(data), {}
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a pandas DataFrame.

        Args:
            df: The DataFrame to filter.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if self.mode == "include":
            return df[df[self.property].isin(self.values)]
        else:
            return df[~df[self.property].isin(self.values)]

    def _filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter a dictionary of data.

        Args:
            data: The dictionary to filter.

        Returns:
            Dict[str, Any]: The filtered dictionary.
        """
        if self.mode == "include":
            return {k: v for k, v in data.items() if v[self.property] in self.values}
        else:
            return {
                k: v for k, v in data.items() if v[self.property] not in self.values
            }
