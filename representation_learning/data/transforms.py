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
    random_state: Optional[int] = Field(
        default=42, description="Random state for reproducible splits"
    )
    stratify_column: Optional[str] = Field(
        default=None, description="Column name to use for stratified splitting"
    )


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


# Register the transform
register_transform(TrainValSplitConfig, TrainValSplitTransform)
