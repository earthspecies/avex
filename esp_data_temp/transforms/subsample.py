import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import field_validator

from ._base import TransformModel

logger = logging.Logger("esp_data")


class SubsampleConfig(TransformModel):
    type: Literal["subsample"]
    property: str
    ratios: dict[str, float]

    # TODO (milad) we support "other" in ratios?

    @field_validator("ratios")
    @classmethod
    def is_in_range(cls, ratios: dict[str, float]) -> dict[str, float]:
        if not all(0 <= r <= 1 for r in ratios.values()):
            raise ValueError("All ratios must be in [0, 1]")
        return ratios


class Subsample:
    """Subsample data based on property ratios."""

    def __init__(self, property: str, ratios: dict[str, float]) -> None:
        self.property = property
        self.ratios = ratios

    @classmethod
    def from_config(cls, cfg: SubsampleConfig) -> "Subsample":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
            return self._subsample_dataframe(data), {}
        # if isinstance(data, dict):
        #     return self._subsample_dict(data), None
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _choose_keys(self, keys: list[Any], ratio: float) -> list[Any]:
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
        groups = []

        for val, ratio in self.ratios.items():
            if val == "other":
                continue
            idx = df.index[df[self.property] == val].tolist()
            chosen = self._choose_keys(idx, ratio)
            groups.append(df.loc[chosen])

        if "other" in self.ratios:
            mask_other = ~df[self.property].isin(self.ratios.keys() - {"other"})
            idx_other = df.index[mask_other].tolist()
            chosen_other = self._choose_keys(idx_other, self.ratios["other"])
            groups.append(df.loc[chosen_other])

        return pd.concat(groups, ignore_index=True)

    # def _subsample_dict(self, data: dict[str, Any]) -> dict[str, Any]:
    #     """Subsample a dictionary of data.

    #     Args:
    #         data: The dictionary to subsample.

    #     Returns:
    #         Dict[str, Any]: The subsampled dictionary.
    #     """
    #     prop = self.cfg.property
    #     ratios = self.cfg.ratios
    #     selected: dict[str, Any] = {}

    #     for val, ratio in ratios.items():
    #         if val == "other":
    #             continue
    #         keys = [k for k, v in data.items() if v[prop] == val]
    #         for k in self._choose_keys(keys, ratio):
    #             selected[k] = data[k]

    #     if "other" in ratios:
    #         other_keys = [
    #             k for k, v in data.items() if v[prop] not in (ratios.keys() - {"other"})
    #         ]
    #         for k in self._choose_keys(other_keys, ratios["other"]):
    #             selected[k] = data[k]

    #     return selected
