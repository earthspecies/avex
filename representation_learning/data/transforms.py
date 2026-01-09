"""Custom transforms for representation learning data processing."""

from typing import Literal, Optional, Tuple

import pandas as pd
from esp_data.transforms import register_transform
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split


class TrainValSplitConfig(BaseModel):
    """Configuration for TrainValSplitTransform.

    This transform splits a dataset into train and validation subsplits
    and returns either the train or validation portion based on the
    subset parameter.
    """

    type: Literal["train_val_split"]
    subset: Literal["train", "validation"] = Field(
        default="train", description="Which subset to return after splitting"
    )
    train_size: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Proportion of the dataset to include in the train split",
    )
    random_state: Optional[int] = Field(default=42, description="Random state for reproducible splits")
    stratify_column: Optional[str] = Field(default=None, description="Column name to use for stratified splitting")


class TrainValSplitTransform:
    """Transform that splits a dataset into train and validation subsplits.

    This transform uses sklearn's train_test_split to divide the input dataset
    into training and validation portions, then returns the requested subset.

    Parameters
    ----------
    subset : Literal["train", "validation"], default="train"
        Which subset to return after splitting
    train_size : float, default=0.8
        Proportion of the dataset to include in the train split
    random_state : Optional[int], default=42
        Random state for reproducible splits
    stratify_column : Optional[str], default=None
        Column name to use for stratified splitting. If provided,
        the split will preserve the proportion of samples for each
        unique value in this column.
    """

    def __init__(
        self,
        subset: Literal["train", "validation"] = "train",
        train_size: float = 0.8,
        random_state: Optional[int] = 42,
        stratify_column: Optional[str] = None,
    ) -> None:
        """Initialize the TrainValSplitTransform.

        Raises
        ------
        ValueError
            If train_size is not between 0 and 1
        """
        self.subset = subset
        self.train_size = train_size
        self.random_state = random_state
        self.stratify_column = stratify_column

        # Validate train_size
        if not 0.0 < train_size < 1.0:
            raise ValueError(f"train_size must be between 0 and 1, got {train_size}")

    @classmethod
    def from_config(cls, cfg: TrainValSplitConfig) -> "TrainValSplitTransform":
        """Create TrainValSplitTransform from configuration.

        Returns
        -------
        TrainValSplitTransform
            Configured transform instance
        """
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Apply the train/validation split transform.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset to split

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            Tuple containing:
            - Transformed data (either train or validation subset)
            - Metadata dictionary with split information

        Raises
        ------
        ValueError
            If stratify_column is specified but not found in data
        """
        if len(data) == 0:
            return data, {
                "subset": self.subset,
                "original_size": 0,
                "split_size": 0,
            }

        # Prepare stratification
        stratify = None
        if self.stratify_column is not None:
            if self.stratify_column not in data.columns:
                raise ValueError(
                    f"Stratify column '{self.stratify_column}' not found in data. "
                    f"Available columns: {list(data.columns)}"
                )
            stratify = data[self.stratify_column]

        # Perform the split
        train_data, val_data = train_test_split(
            data,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        # Select the requested subset
        if self.subset == "train":
            result_data = train_data
        else:  # validation
            result_data = val_data

        # Reset index to ensure continuous indexing
        result_data = result_data.reset_index(drop=True)

        # Prepare metadata
        metadata = {
            "subset": self.subset,
            "original_size": len(data),
            "split_size": len(result_data),
            "train_size": self.train_size,
            "random_state": self.random_state,
            "stratify_column": self.stratify_column,
        }

        return result_data, metadata


class RLUniformSampleConfig(BaseModel):
    """Configuration for RLUniformSampleTransform.

    This transform samples uniformly across a specified property (e.g., label)
    to ensure balanced representation while optionally limiting total samples.
    """

    type: Literal["rl_uniform_sample"]
    property: str = Field(..., description="Property/column name to sample uniformly across (e.g., 'label')")
    ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ratio of samples to keep per unique value in the property column",
    )
    max_samples: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum total number of samples to keep across all classes",
    )
    random_state: Optional[int] = Field(default=42, description="Random state for reproducible sampling")


class RLUniformSampleTransform:
    """Transform that samples uniformly across a specified property.

    This transform ensures balanced representation by sampling the same ratio
    from each unique value in the specified property column. It optionally
    limits the total number of samples.

    Parameters
    ----------
    property : str
        Property/column name to sample uniformly across
    ratio : float, default=1.0
        Ratio of samples to keep per unique value
    max_samples : Optional[int], default=None
        Maximum total number of samples to keep across all classes
    random_state : Optional[int], default=42
        Random state for reproducible sampling
    """

    def __init__(
        self,
        property: str,  # noqa: A002
        ratio: float = 1.0,
        max_samples: Optional[int] = None,
        random_state: Optional[int] = 42,
    ) -> None:
        """Initialize the RLUniformSampleTransform.

        Raises
        ------
        ValueError
            If ratio is not between 0 and 1, or max_samples is not positive
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be between 0 and 1, got {ratio}")
        if max_samples is not None and max_samples < 1:
            raise ValueError(f"max_samples must be >= 1, got {max_samples}")

        self.property = property
        self.ratio = ratio
        self.max_samples = max_samples
        self.random_state = random_state

    @classmethod
    def from_config(cls, cfg: RLUniformSampleConfig) -> "RLUniformSampleTransform":
        """Create RLUniformSampleTransform from configuration.

        Returns
        -------
        RLUniformSampleTransform
            Configured transform instance
        """
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Apply the uniform sampling transform.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset to sample from

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            Tuple containing:
            - Sampled data (DataFrame, not tuple)
            - Metadata dictionary with sampling information

        Raises
        ------
        ValueError
            If property column is not found in data
        """
        if len(data) == 0:
            return data, {
                "uniform_sample": {
                    "original_size": 0,
                    "sampled_size": 0,
                    "property": self.property,
                    "ratio": self.ratio,
                    "max_samples": self.max_samples,
                }
            }

        if self.property not in data.columns:
            raise ValueError(
                f"Property column '{self.property}' not found in data. Available columns: {list(data.columns)}"
            )

        # Get unique values in the property column
        unique_values = data[self.property].unique()
        original_size = len(data)

        # Sample uniformly from each unique value
        sampled_dfs = []
        for value in unique_values:
            value_data = data[data[self.property] == value]
            n_samples = int(len(value_data) * self.ratio)

            if n_samples > 0:
                sampled_value = value_data.sample(
                    n=min(n_samples, len(value_data)),
                    random_state=self.random_state,
                )
                sampled_dfs.append(sampled_value)

        # Combine all sampled data
        if sampled_dfs:
            sampled_data = pd.concat(sampled_dfs, ignore_index=True)
        else:
            sampled_data = pd.DataFrame(columns=data.columns)

        # Apply max_samples limit if specified
        if self.max_samples is not None and len(sampled_data) > self.max_samples:
            # Sample uniformly across all classes to respect max_samples
            samples_per_class = max(1, self.max_samples // len(unique_values))
            final_dfs = []
            for value in unique_values:
                value_data = sampled_data[sampled_data[self.property] == value]
                if len(value_data) > 0:
                    final_value = value_data.sample(
                        n=min(samples_per_class, len(value_data)),
                        random_state=self.random_state,
                    )
                    final_dfs.append(final_value)

            if final_dfs:
                sampled_data = pd.concat(final_dfs, ignore_index=True)
                # If still over limit, randomly sample to exact limit
                if len(sampled_data) > self.max_samples:
                    sampled_data = sampled_data.sample(
                        n=self.max_samples,
                        random_state=self.random_state,
                    ).reset_index(drop=True)
            else:
                sampled_data = pd.DataFrame(columns=data.columns)

        # Shuffle the final result
        sampled_data = sampled_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Prepare metadata
        metadata = {
            "uniform_sample": {
                "original_size": original_size,
                "sampled_size": len(sampled_data),
                "property": self.property,
                "ratio": self.ratio,
                "max_samples": self.max_samples,
                "unique_values_count": len(unique_values),
            }
        }

        # Return DataFrame (not tuple) to ensure compatibility with dataset structure
        return sampled_data, metadata


# Register the transforms
register_transform(TrainValSplitConfig, TrainValSplitTransform)
register_transform(RLUniformSampleConfig, RLUniformSampleTransform)
