"""Temporary dataset configuration classes for training and evaluation.

This module contains configuration classes that depend on esp_data.DatasetConfig.
These are used for training and evaluation workflows, not for the public API.

This is a temporary module that isolates the dataset configuration classes that
require esp_data as a dependency. This allows the main configs.py to be used
by the public API without requiring esp-data to be installed.

Development workflows that rely on run_train.py or run_evaluate.py can
continue to use these configs once the optional dev dependencies are installed.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from esp_data import DatasetConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetCollectionConfig(BaseModel):
    """Configuration for a collection of datasets.

    This is used to define a set of datasets that can be used in training or evaluation.
    It allows specifying multiple datasets with their configurations.

    Attributes
    ----------
    datasets : List[DatasetConfig]
        List of dataset configurations

    concatenate : bool
        If True, concatenate all datasets into a single dataset.
        If False, treat each dataset separately.

    concatenate_method : Literal["hard", "overlap", "soft"]
        Method to use when concatenating datasets:
        'hard' for strict concatenation (all columns must match),
        'overlap' for overlapping columns only,
        'soft' to allow any columns to be present in any dataset.
    """

    train_datasets: Optional[List[DatasetConfig]] = Field(
        None, description="Optional List of training dataset configurations"
    )
    val_datasets: Optional[List[DatasetConfig]] = Field(
        None,
        description="Optional list of validation dataset configurations",
    )
    test_datasets: Optional[List[DatasetConfig]] = Field(
        None,
        description="Optional list of test dataset configurations",
    )
    concatenate_train: bool = Field(
        True,
        description=(
            "If True, concatenate all datasets into a single dataset. If False, treat each dataset separately."
        ),
    )
    concatenate_val: bool = Field(
        True,
        description=(
            "If True, concatenate all evaluation datasets into a single dataset. "
            "If False, treat each evaluation dataset separately."
        ),
    )
    concatenate_test: bool = Field(
        True,
        description=(
            "If True, concatenate all test datasets into a single dataset. "
            "If False, treat each test dataset separately."
        ),
    )
    concatenate_method: Literal["hard", "overlap", "soft"] = Field(
        "soft",
        description=(
            "Method to use when concatenating datasets:"
            "'hard' for strict concatenation (all columns must match),"
            "'overlap' for overlapping columns only,"
            "'soft' to allow any columns to be present in any dataset"
        ),
    )
    transformations: list | None = Field(
        None,
        description=(
            "Optional list of transformations to apply to the concatenated dataset. "
            "These transformations are applied before concatenation."
        ),
    )
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_nonempty_datasets(self) -> "DatasetCollectionConfig":
        # Check that not all of train, val and test are empty
        # one of them has to be provided
        if not (self.train_datasets or self.val_datasets or self.test_datasets):
            raise ValueError("At least one of train_datasets, val_datasets,or test_datasets must be provided.")
        return self


class EvaluationSet(BaseModel):
    """Configuration for a single evaluation set (train/val/test triplet)."""

    name: str = Field(
        ...,
        description="Name of this evaluation set (e.g., 'dog_classification')",
    )
    train: DatasetConfig = Field(..., description="Training dataset configuration")
    validation: DatasetConfig = Field(..., description="Validation dataset configuration")
    test: DatasetConfig = Field(..., description="Test dataset configuration")
    metrics: List[str] = Field(
        default_factory=lambda: ["accuracy"],
        description="List of metrics to compute for this evaluation set",
    )

    # Add retrieval mode configuration
    retrieval_mode: Literal["test_vs_test", "train_vs_test"] = Field(
        "test_vs_test",
        description=("Retrieval evaluation mode: 'test_vs_test' (current default) or 'train_vs_test'"),
    )

    model_config = ConfigDict(extra="forbid")

    def to_dataset_collection_config(self) -> DatasetCollectionConfig:
        """Convert this evaluation set to a DatasetCollectionConfig.

        Returns
        -------
        DatasetCollectionConfig
            A config that can be used with esp-data's dataset loading functionality
        """
        return DatasetCollectionConfig(
            train_datasets=[self.train],
            val_datasets=[self.validation],
            test_datasets=[self.test],
            concatenate_train=True,
            concatenate_val=True,
            concatenate_test=True,
            concatenate_method="soft",
        )


class BenchmarkEvaluationConfig(BaseModel):
    """Configuration for benchmark evaluation wrapping
    esp-data's DatasetCollectionConfig for actual data loading.

    Example
    -------
    ```yaml
    benchmark_name: "bioacoustic_benchmark_v1"
    evaluation_sets:
      - name: "dog_classification"
        train:
          dataset_name: beans
          split: dogs_train
          type: classification
          # ... other config
        validation:
          dataset_name: beans
          split: dogs_validation
          type: classification
          # ... other config
        test:
          dataset_name: beans
          split: dogs_test
          type: classification
          # ... other config
        metrics: [accuracy, balanced_accuracy]
    ```
    """

    benchmark_name: str = Field(..., description="Name of this benchmark")
    evaluation_sets: List[EvaluationSet] = Field(
        ...,
        description=("List of evaluation sets (train/val/test triplets) in this benchmark"),
    )

    model_config = ConfigDict(extra="forbid")

    def get_evaluation_set(self, name: str) -> EvaluationSet:
        """Get a specific evaluation set by name.

        Parameters
        ----------
        name : str
            Name of the evaluation set to retrieve

        Returns
        -------
        EvaluationSet
            The requested evaluation set

        Raises
        ------
        ValueError
            If no evaluation set with the given name is found
        """
        for eval_set in self.evaluation_sets:
            if eval_set.name == name:
                return eval_set
        raise ValueError(f"No evaluation set named '{name}' found in benchmark '{self.benchmark_name}'")

    def get_all_evaluation_sets(
        self,
    ) -> List[Tuple[str, DatasetCollectionConfig]]:
        """Get all evaluation sets as (name, DatasetCollectionConfig) pairs.

        This is the main interface for evaluation loops - it provides each evaluation
        set converted to the format needed by esp-data for actual data loading.

        Returns
        -------
        List[Tuple[str, DatasetCollectionConfig]]
            List of (evaluation_set_name, dataset_collection_config) pairs
        """
        return [(eval_set.name, eval_set.to_dataset_collection_config()) for eval_set in self.evaluation_sets]

    def get_metrics_for_evaluation_set(self, name: str) -> List[str]:
        """Get the metrics list for a specific evaluation set.

        Parameters
        ----------
        name : str
            Name of the evaluation set

        Returns
        -------
        List[str]
            List of metric names for this evaluation set
        """
        return self.get_evaluation_set(name).metrics
