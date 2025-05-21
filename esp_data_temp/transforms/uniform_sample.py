import logging
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from . import register_transform

logger = logging.Logger("esp_data")


class UniformSampleConfig(BaseModel):
    type: Literal["uniform_sample"]
    property: str
    ratio: float

    @field_validator("ratio")
    @classmethod
    def is_in_range(cls, ratio: float) -> float:
        if not 0 <= ratio <= 1:
            raise ValueError("Ratio must be in [0, 1]")
        return ratio


# TODO (Gagan) I'm a bit confused by what UniformSample is ... it sounds like a subset
# of Subsample just that the ratios "fixed" internally to make the dataset uniformly
# distributed across a certain property. But here a scalar "ratio" is being provided
# which would apply to all the unique groups in a property, and hence will not
# distribute uniformly.


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
            # TODO is this the right way to set up the random seed? Do we want to fix it
            # here?
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


register_transform(UniformSampleConfig, UniformSample)
