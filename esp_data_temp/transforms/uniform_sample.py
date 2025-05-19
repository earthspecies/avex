import logging
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import field_validator

from ._base import TransformModel

logger = logging.Logger("esp_data")


class UniformSampleConfig(TransformModel):
    type: Literal["uniform_sample"]
    property: str
    ratio: float

    @field_validator("ratio")
    @classmethod
    def is_in_range(cls, ratio: float) -> float:
        if not 0 <= ratio <= 1:
            raise ValueError("Ratio must be in [0, 1]")
        return ratio


class UniformSample:
    """Uniformly sample data based on a property."""

    def __init__(self, property: str, ratio: float) -> None:
        self.property = property
        self.ratio = ratio

    @classmethod
    def from_config(cls, cfg: UniformSampleConfig) -> "UniformSample":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if isinstance(data, pd.DataFrame):
            return self._uniform_sample_dataframe(data), {}
        # if isinstance(data, dict):
        #     return self._uniform_sample_dict(data)
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _uniform_sample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Uniformly sample a pandas DataFrame.

        Returns:
        --------
            pd.DataFrame: A uniformly sampled DataFrame.

        """

        # Group by the property and sample uniformly
        groups = []
        for _, group in df.groupby(self.property):
            n_samples = max(1, int(len(group) * self.ratio))
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(group), size=n_samples, replace=False)
            groups.append(group.iloc[sampled_indices])

        return pd.concat(groups, ignore_index=True)

    # def _uniform_sample_dict(self, data: dict[str, Any]) -> dict[str, Any]:
    #     """
    #     Uniformly sample a dictionary of data.

    #     """
    #     prop = self.cfg.property
    #     ratio = self.cfg.ratio
    #     selected: dict[str, Any] = {}

    #     # Group by the property
    #     groups: dict[str, list[str]] = {}
    #     for k, v in data.items():
    #         val = v[prop]
    #         if val not in groups:
    #             groups[val] = []
    #         groups[val].append(k)

    #     # Sample uniformly from each group
    #     for keys in groups.values():
    #         n_samples = max(1, int(len(keys) * ratio))
    #         rng = np.random.default_rng(seed=42)
    #         sampled_keys = rng.choice(keys, size=n_samples, replace=False)
    #         for k in sampled_keys:
    #             selected[k] = data[k]

    #     return selected
