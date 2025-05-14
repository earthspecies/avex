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

    def __init__(self, config: FilterConfig) -> None:
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
        """Filter the data based on property values.

        Args:
            data: The data to filter (DataFrame or dict).

        Returns:
            The filtered data (same type as input).

        Raises:
            TypeError: If the data type is not supported.
        """
        if isinstance(data, pd.DataFrame):
            return self._filter_dataframe(data), None
        elif isinstance(data, dict):
            return self._filter_dict(data), None
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a pandas DataFrame.

        Args:
            df: The DataFrame to filter.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if self.config.mode == "include":
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
