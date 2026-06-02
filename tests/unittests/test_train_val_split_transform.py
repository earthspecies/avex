"""Unit tests for TrainValSplitTransform.

These tests require esp_data which is an internal dependency.
They are skipped when esp_data is not installed.
"""

import pytest

# Skip entire module if esp_data is not installed (internal dependency)
# Must be before imports that trigger esp_data loading (e.g., avex.data.transforms)
pytest.importorskip("esp_data")

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from esp_data.backends import DataBackend, PandasBackend, PolarsBackend  # noqa: E402

from avex.data.transforms import (  # noqa: E402
    TrainValSplitConfig,
    TrainValSplitTransform,
)


@pytest.fixture(params=[PandasBackend, PolarsBackend], ids=["pandas", "polars"])
def backend_cls(request: pytest.FixtureRequest) -> type[DataBackend]:
    """Run each test against both esp_data backends.

    Datasets are always loaded through an esp_data backend (pandas or polars)
    before transforms run, so the transform operates on a ``DataBackend`` rather
    than a raw DataFrame. Parametrizing here exercises both backend paths through
    the same protocol the transform relies on.

    Returns
    -------
    type[DataBackend]
        The backend class for the current parametrization.
    """
    return request.param


def _wrap(backend_cls: type[DataBackend], columns: dict) -> DataBackend:
    """Wrap a column dict in the given backend (pandas or polars).

    Returns
    -------
    DataBackend
        Backend instance wrapping the columns.
    """
    frame = pd.DataFrame(columns) if backend_cls is PandasBackend else pl.DataFrame(columns)
    return backend_cls(frame)


def _ids(result: DataBackend, column: str = "id") -> set:
    """Collect a column's values backend-agnostically via row iteration.

    Returns
    -------
    set
        The set of values found in ``column``.
    """
    return {row[column] for row in result}


class TestTrainValSplitTransform:
    """Test cases for TrainValSplitTransform."""

    def test_basic_split_train(self, backend_cls: type[DataBackend]) -> None:
        """Test basic train split functionality."""
        data = _wrap(
            backend_cls,
            {
                "path": [f"data/sample_{i}.wav" for i in range(100)],
                "label": [str(i % 5) for i in range(100)],  # 5 different labels
                "text": [f"Sample text {i}" for i in range(100)],
            },
        )

        config = TrainValSplitConfig(
            type="train_val_split",
            subset="train",
            train_size=0.8,
            random_state=42,
        )

        transform = TrainValSplitTransform.from_config(config)
        result_data, metadata = transform(data)

        # Verify results
        assert len(result_data) == 80  # 80% of 100
        assert metadata["subset"] == "train"
        assert metadata["original_size"] == 100
        assert metadata["split_size"] == 80
        assert metadata["train_size"] == 0.8
        assert metadata["random_state"] == 42

        # Verify columns are preserved
        assert list(result_data.columns) == ["path", "label", "text"]

    def test_basic_split_validation(self, backend_cls: type[DataBackend]) -> None:
        """Test basic validation split functionality."""
        data = _wrap(
            backend_cls,
            {
                "path": [f"data/sample_{i}.wav" for i in range(100)],
                "label": [str(i % 5) for i in range(100)],
                "text": [f"Sample text {i}" for i in range(100)],
            },
        )

        config = TrainValSplitConfig(
            type="train_val_split",
            subset="validation",
            train_size=0.8,
            random_state=42,
        )

        transform = TrainValSplitTransform.from_config(config)
        result_data, metadata = transform(data)

        # Verify results
        assert len(result_data) == 20  # 20% of 100
        assert metadata["subset"] == "validation"
        assert metadata["original_size"] == 100
        assert metadata["split_size"] == 20

    def test_stratified_split(self, backend_cls: type[DataBackend]) -> None:
        """Test stratified splitting functionality."""
        data = _wrap(
            backend_cls,
            {
                "path": [f"data/sample_{i}.wav" for i in range(100)],
                "label": ["A"] * 60 + ["B"] * 30 + ["C"] * 10,  # Imbalanced
                "text": [f"Sample text {i}" for i in range(100)],
            },
        )

        config = TrainValSplitConfig(
            type="train_val_split",
            subset="train",
            train_size=0.8,
            random_state=42,
            stratify_column="label",
        )

        transform = TrainValSplitTransform.from_config(config)
        result_data, metadata = transform(data)

        # Verify stratification maintained proportions
        train_label_counts = result_data.histogram("label")

        assert train_label_counts["A"] == 48
        assert train_label_counts["B"] == 24
        assert train_label_counts["C"] == 8

        assert metadata["stratify_column"] == "label"

    def test_empty_data(self, backend_cls: type[DataBackend]) -> None:
        """Test handling of empty dataset."""
        data = _wrap(backend_cls, {"path": [], "label": [], "text": []})

        config = TrainValSplitConfig(type="train_val_split", subset="train", train_size=0.8)

        transform = TrainValSplitTransform.from_config(config)
        result_data, metadata = transform(data)

        assert len(result_data) == 0
        assert metadata["original_size"] == 0
        assert metadata["split_size"] == 0

    def test_different_train_sizes(self, backend_cls: type[DataBackend]) -> None:
        """Test different train size configurations."""
        data = _wrap(
            backend_cls,
            {
                "path": [f"data/sample_{i}.wav" for i in range(100)],
                "label": [str(i % 2) for i in range(100)],
            },
        )

        config = TrainValSplitConfig(
            type="train_val_split",
            subset="train",
            train_size=0.9,
            random_state=42,
        )

        transform = TrainValSplitTransform.from_config(config)
        result_data, metadata = transform(data)

        assert len(result_data) == 90
        assert metadata["train_size"] == 0.9

    def test_reproducibility(self, backend_cls: type[DataBackend]) -> None:
        """Test that the same random_state produces identical results."""
        columns = {
            "path": [f"data/sample_{i}.wav" for i in range(100)],
            "label": [str(i % 3) for i in range(100)],
        }

        config = TrainValSplitConfig(
            type="train_val_split",
            subset="train",
            train_size=0.7,
            random_state=123,
        )

        # Apply transform twice over independently wrapped copies
        result1, _ = TrainValSplitTransform.from_config(config)(_wrap(backend_cls, columns))
        result2, _ = TrainValSplitTransform.from_config(config)(_wrap(backend_cls, columns))

        # Results should be identical (compare backend-agnostically via rows)
        assert list(result1) == list(result2)

    def test_invalid_train_size(self) -> None:
        """Test validation of train_size parameter."""
        with pytest.raises(ValueError, match="train_size must be between 0 and 1"):
            TrainValSplitTransform(train_size=1.5)

        with pytest.raises(ValueError, match="train_size must be between 0 and 1"):
            TrainValSplitTransform(train_size=0.0)

    def test_invalid_stratify_column(self, backend_cls: type[DataBackend]) -> None:
        """Test error handling for invalid stratify column."""
        data = _wrap(
            backend_cls,
            {
                "path": [f"data/sample_{i}.wav" for i in range(10)],
                "label": [str(i % 2) for i in range(10)],
            },
        )

        config = TrainValSplitConfig(
            type="train_val_split",
            subset="train",
            stratify_column="nonexistent_column",
        )

        transform = TrainValSplitTransform.from_config(config)

        with pytest.raises(ValueError, match="Stratify column 'nonexistent_column' not found"):
            transform(data)

    def test_validation_subset_complement(self, backend_cls: type[DataBackend]) -> None:
        """Test that train and validation subsets are complementary."""
        columns = {
            "path": [f"data/sample_{i}.wav" for i in range(50)],
            "label": [str(i % 3) for i in range(50)],
            "id": list(range(50)),  # Unique identifier for each row
        }

        # Get train subset
        train_config = TrainValSplitConfig(
            type="train_val_split",
            subset="train",
            train_size=0.8,
            random_state=42,
        )
        train_data, _ = TrainValSplitTransform.from_config(train_config)(_wrap(backend_cls, columns))

        # Get validation subset
        val_config = TrainValSplitConfig(
            type="train_val_split",
            subset="validation",
            train_size=0.8,
            random_state=42,
        )
        val_data, _ = TrainValSplitTransform.from_config(val_config)(_wrap(backend_cls, columns))

        # Verify they are complementary (no overlap, together they form the original)
        # Use the unique identifier column to check overlap since indices are reset
        train_ids = _ids(train_data)
        val_ids = _ids(val_data)
        original_ids = set(columns["id"])

        assert len(train_ids.intersection(val_ids)) == 0  # No overlap
        assert train_ids.union(val_ids) == original_ids  # Together form original
        assert len(train_data) + len(val_data) == len(columns["id"])  # Sizes add up
