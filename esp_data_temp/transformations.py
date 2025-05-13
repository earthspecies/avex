"""
Data transformation utilities for filtering and subsampling datasets.
"""

import logging
from collections.abc import Callable
from functools import partial
from typing import Annotated, Any, Dict, List, Union, get_args

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Literal

logger = logging.Logger("esp_data")

# TODO (milad) find a way to use RegisteredTransformConfigs in the type hint below?
_TRANSFORM_REGISTRY: dict[str, type["TransformModel"]] = {}
RegisteredTransformConfigs = list[Any]


class TransformModel(BaseModel):
    """
    Base class for all transform configurations.

    All transform configurations should inherit from this class and define a unique
    `type` attribute. This class does the registration of subclasses in the
    `_TRANSFORM_REGISTRY` dictionary. The `RegisteredTransformConfigs` class variable is
    a union of all registered transform types, allowing for easy validation and type
    checking. The `type` attribute is used as a discriminator for the union type.
    """

    # TODO (milad) I wonder if this can be done with a simple decorator. Decorators are
    # also called when a class is defined (and not when they're instantiated). Having
    # a base class allows us to define the common "type" field but my type checker is
    # complaining that you can't have the base hint as "str" while child classes are
    # using Literal["..."].

    type: str

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs) -> None:
        """
        This method is called when a subclass of TransformModel is created. It registers
        the subclass in the `_TRANSFORM_REGISTRY` dictionary using the `type` attribute
        as the key. It also rebuilds the `RegisteredTransformConfigs` union type to
        include the new subclass, which is used for instantiating a _list_ of transforms

        Raises
        ------
        ValueError
            If the type is already registered, a ValueError is raised.
        """

        # See this issue below to understand we need `__pydantic_init_subclass__`
        # instead of `__init_subclass__`:
        # https://github.com/pydantic/pydantic/issues/6854

        # TODO (milad) type checkers do not like dynamic types and complain about
        # RegisteredTransformConfigs. This type is obviously useful for Pydantic checks
        # but investigate if there's a static option.

        type_vals = get_args(cls.model_fields["type"].annotation)
        if len(type_vals) == 1:
            v = type_vals[0]

            if v in _TRANSFORM_REGISTRY:
                raise ValueError(
                    f"Transform type '{v}' is already registered. "
                    "Please use a unique type name."
                )

            _TRANSFORM_REGISTRY[v] = cls
            TransformModel._rebuild_union_type()

    @classmethod
    def _rebuild_union_type(cls):
        global RegisteredTransformConfigs
        RegisteredTransformConfigs = Annotated[
            Union[tuple(_TRANSFORM_REGISTRY.values())], Field(discriminator="type")
        ]


class FilterConfig(TransformModel):
    type: Literal["filter"]
    mode: Literal["include", "exclude"] = "include"
    property: str
    values: list[str]


class SubsampleConfig(TransformModel):
    type: Literal["subsample"]
    property: str
    ratios: dict[str, float]

    # TODO (milad) add a validator that does below:
    #         if not all(0 <= r <= 1 for r in config.ratios.values()):
    # raise ValueError("All ratios must be in [0, 1]")


class UniformSampleConfig(TransformModel):
    type: Literal["uniform_sample"]
    property: str
    ratio: float


class LabelFromFeature(TransformModel):
    type: Literal["label_from_feature"]
    feature: str
    num_classes: int | Literal["auto"] = "auto"
    output_feature: str = "label"
    override: bool = False


# TODO (milad) I kind what all these transform functions to have a simple interface so
# maybe ban positional arguments once you've edited them all


# TODO (milad) name too similar too config class
# @register_transform(LabelFromFeature)
def create_labels(
    df: pd.DataFrame, cfg: LabelFromFeature
) -> tuple[pd.DataFrame, dict | None]:
    if cfg.output_feature in df and not cfg.override:
        assert False, "TODO (milad)"

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


class Filter:
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


class Subsample:
    """Subsample data based on property ratios."""

    def __init__(self, config: SubsampleConfig):
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
            return self._subsample_dataframe(data), None
        if isinstance(data, dict):
            return self._subsample_dict(data), None
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


class UniformSample:
    """Uniformly sample data based on a property."""

    def __init__(self, config: UniformSampleConfig) -> None:
        # Pydantic enforcement guarantees `config.type == 'uniform_sample'`.
        if not 0 <= config.ratio <= 1:
            raise ValueError("Ratio must be in [0, 1]")
        self.cfg = config

    def __call__(self, data: Union[pd.DataFrame, Dict[str, Any]]):  # noqa: ANN201
        if isinstance(data, pd.DataFrame):
            return self._uniform_sample_dataframe(data)
        if isinstance(data, dict):
            return self._uniform_sample_dict(data)
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _uniform_sample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Uniformly sample a pandas DataFrame."""
        prop = self.cfg.property
        ratio = self.cfg.ratio

        # Group by the property and sample uniformly
        groups = []
        for _, group in df.groupby(prop):
            n_samples = max(1, int(len(group) * ratio))
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(group), size=n_samples, replace=False)
            groups.append(group.iloc[sampled_indices])

        return pd.concat(groups, ignore_index=True)

    def _uniform_sample_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Uniformly sample a dictionary of data."""
        prop = self.cfg.property
        ratio = self.cfg.ratio
        selected: Dict[str, Any] = {}

        # Group by the property
        groups: Dict[str, List[str]] = {}
        for k, v in data.items():
            val = v[prop]
            if val not in groups:
                groups[val] = []
            groups[val].append(k)

        # Sample uniformly from each group
        for keys in groups.values():
            n_samples = max(1, int(len(keys) * ratio))
            rng = np.random.default_rng(seed=42)
            sampled_keys = rng.choice(keys, size=n_samples, replace=False)
            for k in sampled_keys:
                selected[k] = data[k]

        return selected


def build_transforms(
    transform_configs: list[RegisteredTransformConfigs],
) -> list[Callable[[pd.DataFrame], pd.DataFrame]]:
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
