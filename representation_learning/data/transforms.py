"""
Custom data transforms for preprocessing.
Used to extend esp_data transforms with application-specific functionality.
"""

import logging
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger("representation_learning.transforms")


class FilterNoneLabelsConfig(BaseModel):
    """Configuration for filtering out 'None' labels from lists."""

    type: Literal["filter_none_labels"]
    input_feature: str
    output_feature: str | None = None
    none_values: list[str] = ["None", "none", "NONE"]
    override: bool = False


class FilterNoneLabels:
    """
    Transform that removes 'None' values from label lists.

    This is useful for detection datasets where 'None' represents the absence
    of any animal sound rather than a trainable class. Converts ['None'] to []
    and ['Species1', 'None', 'Species2'] to ['Species1', 'Species2'].

    Parameters
    ----------
    input_feature : str
        The name of the column containing label lists
    output_feature : str, optional
        The name of the output column. If None, overwrites the input column.
    none_values : list[str], default=["None", "none", "NONE"]
        List of string values to treat as "None" and filter out
    override : bool, default=False
        Whether to override the output column if it already exists

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "labels": [["None"], ["Species1"], ["Species1", "None", "Species2"], []]
    ... })
    >>> config = FilterNoneLabelsConfig(
    ...     type="filter_none_labels",
    ...     input_feature="labels",
    ...     output_feature="filtered_labels"
    ... )
    >>> transform = FilterNoneLabels.from_config(config)
    >>> result_df, metadata = transform(df)
    >>> print(result_df["filtered_labels"].tolist())
    [[], ['Species1'], ['Species1', 'Species2'], []]
    """

    def __init__(
        self,
        *,
        input_feature: str,
        output_feature: str | None = None,
        none_values: list[str] = None,
        override: bool = False,
    ):
        self.input_feature = input_feature
        self.output_feature = output_feature or input_feature
        self.none_values = none_values or ["None", "none", "NONE"]
        self.override = override

    @classmethod
    def from_config(cls, cfg: FilterNoneLabelsConfig) -> "FilterNoneLabels":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if self.output_feature in df and not self.override:
            raise ValueError(
                f"Feature '{self.output_feature}' already exists in DataFrame. "
                "Set `override=True` to replace it."
            )

        def _filter_none_labels(value: Any) -> list[str]:
            """Filter out None values from label lists."""
            # Handle pandas NaN/None for single values
            if not isinstance(value, list) and pd.isna(value):
                return []
            if not isinstance(value, list):
                # Single value - check if it's a None value
                if isinstance(value, str) and value in self.none_values:
                    return []
                return [str(value)]

            # Filter out None values from the list
            filtered = [item for item in value if item not in self.none_values]
            return filtered

        # Count statistics before transformation
        original_none_count = 0

        for orig in df[self.input_feature]:
            if isinstance(orig, list):
                original_none_count += sum(
                    1 for item in orig if item in self.none_values
                )
            elif isinstance(orig, str) and orig in self.none_values:
                original_none_count += 1

        # Apply the transformation
        df = df.copy()
        df[self.output_feature] = df[self.input_feature].apply(_filter_none_labels)

        logger.info(
            f"Filtered {original_none_count} 'None' labels from column '{self.input_feature}' "
            f"into column '{self.output_feature}'"
        )

        metadata = {
            "input_feature": self.input_feature,
            "output_feature": self.output_feature,
            "none_values_filtered": self.none_values,
            "none_count_removed": int(original_none_count),
        }

        return df, metadata


class SplitCommaSeparatedConfig(BaseModel):
    """Configuration for splitting comma-separated strings into lists."""

    type: Literal["split_comma_separated"]
    input_feature: str
    output_feature: str | None = None
    strip_whitespace: bool = True
    override: bool = False


class SplitCommaSeparated:
    """
    Transform that splits comma-separated strings into lists of individual items.

    This is useful for preprocessing multi-label data where labels are stored as
    comma-separated strings (e.g., "Scaly-naped pigeon, Hedrick's coqui") before
    applying the multilabel_from_features transform.

    Parameters
    ----------
    input_feature : str
        The name of the column containing comma-separated strings
    output_feature : str, optional
        The name of the output column. If None, overwrites the input column.
    strip_whitespace : bool, default=True
        Whether to strip whitespace from individual items after splitting
    override : bool, default=False
        Whether to override the output column if it already exists

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "labels": ["cat, dog", "bird", "cat, mouse, bird"]
    ... })
    >>> config = SplitCommaSeparatedConfig(
    ...     type="split_comma_separated",
    ...     input_feature="labels",
    ...     output_feature="label_list"
    ... )
    >>> transform = SplitCommaSeparated.from_config(config)
    >>> result_df, metadata = transform(df)
    >>> print(result_df["label_list"].tolist())
    [['cat', 'dog'], ['bird'], ['cat', 'mouse', 'bird']]
    """

    def __init__(
        self,
        *,
        input_feature: str,
        output_feature: str | None = None,
        strip_whitespace: bool = True,
        override: bool = False,
    ):
        self.input_feature = input_feature
        self.output_feature = output_feature or input_feature
        self.strip_whitespace = strip_whitespace
        self.override = override

    @classmethod
    def from_config(cls, cfg: SplitCommaSeparatedConfig) -> "SplitCommaSeparated":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if self.output_feature in df and not self.override:
            raise ValueError(
                f"Feature '{self.output_feature}' already exists in DataFrame. "
                "Set `override=True` to replace it."
            )

        def _split_comma_separated(value: Any) -> list[str]:
            """Split comma-separated string into list of strings."""
            if pd.isna(value):
                return []
            if isinstance(value, list):
                # Already a list, return as-is
                return value
            if not isinstance(value, str):
                # Single non-string value, convert to string and wrap in list
                return [str(value)]

            # Split comma-separated string
            items = value.split(",")
            if self.strip_whitespace:
                items = [item.strip() for item in items]

            # Remove empty strings
            items = [item for item in items if item]

            return items

        # Apply the transformation
        df = df.copy()
        df[self.output_feature] = df[self.input_feature].apply(_split_comma_separated)

        # Count statistics
        original_strings = (
            df[self.input_feature]
            .apply(lambda x: isinstance(x, str) and "," in str(x))
            .sum()
        )

        logger.info(
            f"Split {original_strings} comma-separated strings in column '{self.input_feature}' "
            f"into lists in column '{self.output_feature}'"
        )

        metadata = {
            "input_feature": self.input_feature,
            "output_feature": self.output_feature,
            "comma_separated_count": int(original_strings),
        }

        return df, metadata


class ExcludeEmptyLabelsConfig(BaseModel):
    """Configuration for excluding examples with empty label lists."""

    type: Literal["exclude_empty_labels"]
    label_feature: str
    min_labels: int = 1


class ExcludeEmptyLabels:
    """
    Transform that excludes entire examples with empty label lists.

    This is useful for evaluation contexts where examples without any labels
    should be completely excluded from the dataset, such as BirdSet detection
    tasks during test evaluation.

    Parameters
    ----------
    label_feature : str
        The name of the column containing label lists to check
    min_labels : int, default=1
        Minimum number of labels required to keep an example

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "labels_as_list": [['bird'], [], ['cat', 'dog'], []],
    ...     "audio_path": ["a.wav", "b.wav", "c.wav", "d.wav"]
    ... })
    >>> config = ExcludeEmptyLabelsConfig(
    ...     type="exclude_empty_labels",
    ...     label_feature="labels_as_list"
    ... )
    >>> transform = ExcludeEmptyLabels.from_config(config)
    >>> result_df, metadata = transform(df)
    >>> print(len(result_df))  # Should be 2 (only examples with labels)
    2
    >>> print(result_df["labels_as_list"].tolist())
    [['bird'], ['cat', 'dog']]
    """

    def __init__(
        self,
        *,
        label_feature: str,
        min_labels: int = 1,
    ):
        self.label_feature = label_feature
        self.min_labels = min_labels

    @classmethod
    def from_config(cls, cfg: ExcludeEmptyLabelsConfig) -> "ExcludeEmptyLabels":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if self.label_feature not in df.columns:
            raise ValueError(
                f"Label feature '{self.label_feature}' not found in DataFrame"
            )

        original_count = len(df)

        # Create mask for examples with sufficient labels
        def has_sufficient_labels(labels):
            if pd.isna(labels):
                return False
            if not isinstance(labels, list):
                # Single label - count as 1 if not None/empty
                return bool(labels) and str(labels).strip() != ""
            # List of labels - check if we have enough non-empty labels
            non_empty_labels = [
                label for label in labels if label and str(label).strip() != ""
            ]
            return len(non_empty_labels) >= self.min_labels

        mask = df[self.label_feature].apply(has_sufficient_labels)

        # Filter the dataframe
        df_filtered = df[mask].copy().reset_index(drop=True)

        excluded_count = original_count - len(df_filtered)

        logger.info(
            f"Excluded {excluded_count} examples with insufficient labels from column '{self.label_feature}' "
            f"(minimum required: {self.min_labels}). Kept {len(df_filtered)}/{original_count} examples."
        )

        metadata = {
            "label_feature": self.label_feature,
            "min_labels": self.min_labels,
            "original_count": int(original_count),
            "excluded_count": int(excluded_count),
            "final_count": int(len(df_filtered)),
        }

        return df_filtered, metadata


# Register custom transforms with esp_data registry
try:
    from esp_data.transforms import register_transform

    register_transform(FilterNoneLabelsConfig, FilterNoneLabels)
    register_transform(SplitCommaSeparatedConfig, SplitCommaSeparated)
    register_transform(ExcludeEmptyLabelsConfig, ExcludeEmptyLabels)

    logger.info(
        "Successfully registered custom transforms: filter_none_labels, split_comma_separated, exclude_empty_labels"
    )

except ImportError as e:
    logger.warning(f"Could not register custom transforms with esp_data: {e}")
except Exception as e:
    logger.error(f"Error registering custom transforms: {e}")
