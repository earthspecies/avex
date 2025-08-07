"""Unit tests for TrainValSplitTransform."""

import pandas as pd
import pytest

from representation_learning.data.transforms import (
    TrainValSplitConfig,
    TrainValSplitTransform,
)


class TestTrainValSplitTransform:
    """Test cases for TrainValSplitTransform."""

    def test_basic_split_train(self) -> None:
        """Test basic train split functionality."""
        # Create sample data
        data = pd.DataFrame(
            {
                "path": [f"data/sample_{i}.wav" for i in range(100)],
                "label": [str(i % 5) for i in range(100)],  # 5 different labels
                "text": [f"Sample text {i}" for i in range(100)],
            }
        )

        # Create transform config for train subset
        config = TrainValSplitConfig(
            type="train_val_split", subset="train", train_size=0.8, random_state=42
        )

        # Create and apply transform
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

        # Verify index is reset
        assert result_data.index.tolist() == list(range(80))

    def test_basic_split_validation(self) -> None:
        """Test basic validation split functionality."""
        # Create sample data
        data = pd.DataFrame(
            {
                "path": [f"data/sample_{i}.wav" for i in range(100)],
                "label": [str(i % 5) for i in range(100)],
                "text": [f"Sample text {i}" for i in range(100)],
            }
        )

        # Create transform config for validation subset
        config = TrainValSplitConfig(
            type="train_val_split", subset="validation", train_size=0.8, random_state=42
        )

        # Create and apply transform
        transform = TrainValSplitTransform.from_config(config)
        result_data, metadata = transform(data)

        # Verify results
        assert len(result_data) == 20  # 20% of 100
        assert metadata["subset"] == "validation"
        assert metadata["original_size"] == 100
        assert metadata["split_size"] == 20

    def test_stratified_split(self) -> None:
        """Test stratified splitting functionality."""
        # Create sample data with imbalanced labels
        data = pd.DataFrame(
            {
                "path": [f"data/sample_{i}.wav" for i in range(100)],
                "label": ["A"] * 60 + ["B"] * 30 + ["C"] * 10,  # Imbalanced
                "text": [f"Sample text {i}" for i in range(100)],
            }
        )

        # Create transform config with stratification
        config = TrainValSplitConfig(
            type="train_val_split",
            subset="train",
            train_size=0.8,
            random_state=42,
            stratify_column="label",
        )

        # Create and apply transform
        transform = TrainValSplitTransform.from_config(config)
        result_data, metadata = transform(data)

        # Verify stratification maintained proportions
        train_label_counts = result_data["label"].value_counts().sort_index()

        # Compare values, not exact Series (to avoid index name issues)
        assert train_label_counts["A"] == 48
        assert train_label_counts["B"] == 24
        assert train_label_counts["C"] == 8

        assert metadata["stratify_column"] == "label"

    def test_empty_data(self) -> None:
        """Test handling of empty dataset."""
        data = pd.DataFrame(columns=["path", "label", "text"])

        config = TrainValSplitConfig(
            type="train_val_split", subset="train", train_size=0.8
        )

        transform = TrainValSplitTransform.from_config(config)
        result_data, metadata = transform(data)

        assert len(result_data) == 0
        assert metadata["original_size"] == 0
        assert metadata["split_size"] == 0

    def test_different_train_sizes(self) -> None:
        """Test different train size configurations."""
        data = pd.DataFrame(
            {
                "path": [f"data/sample_{i}.wav" for i in range(100)],
                "label": [str(i % 2) for i in range(100)],
            }
        )

        # Test 90/10 split
        config = TrainValSplitConfig(
            type="train_val_split", subset="train", train_size=0.9, random_state=42
        )

        transform = TrainValSplitTransform.from_config(config)
        result_data, metadata = transform(data)

        assert len(result_data) == 90
        assert metadata["train_size"] == 0.9

    def test_reproducibility(self) -> None:
        """Test that the same random_state produces identical results."""
        data = pd.DataFrame(
            {
                "path": [f"data/sample_{i}.wav" for i in range(100)],
                "label": [str(i % 3) for i in range(100)],
            }
        )

        config = TrainValSplitConfig(
            type="train_val_split", subset="train", train_size=0.7, random_state=123
        )

        # Apply transform twice
        transform1 = TrainValSplitTransform.from_config(config)
        result1, _ = transform1(data)

        transform2 = TrainValSplitTransform.from_config(config)
        result2, _ = transform2(data)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_invalid_train_size(self) -> None:
        """Test validation of train_size parameter."""
        with pytest.raises(ValueError, match="train_size must be between 0 and 1"):
            TrainValSplitTransform(train_size=1.5)

        with pytest.raises(ValueError, match="train_size must be between 0 and 1"):
            TrainValSplitTransform(train_size=0.0)

    def test_invalid_stratify_column(self) -> None:
        """Test error handling for invalid stratify column."""
        data = pd.DataFrame(
            {
                "path": [f"data/sample_{i}.wav" for i in range(10)],
                "label": [str(i % 2) for i in range(10)],
            }
        )

        config = TrainValSplitConfig(
            type="train_val_split", subset="train", stratify_column="nonexistent_column"
        )

        transform = TrainValSplitTransform.from_config(config)

        with pytest.raises(
            ValueError, match="Stratify column 'nonexistent_column' not found"
        ):
            transform(data)

    def test_validation_subset_complement(self) -> None:
        """Test that train and validation subsets are complementary."""
        data = pd.DataFrame(
            {
                "path": [f"data/sample_{i}.wav" for i in range(50)],
                "label": [str(i % 3) for i in range(50)],
                "id": list(range(50)),  # Unique identifier for each row
            }
        )

        # Get train subset
        train_config = TrainValSplitConfig(
            type="train_val_split", subset="train", train_size=0.8, random_state=42
        )
        train_transform = TrainValSplitTransform.from_config(train_config)
        train_data, _ = train_transform(data)

        # Get validation subset
        val_config = TrainValSplitConfig(
            type="train_val_split", subset="validation", train_size=0.8, random_state=42
        )
        val_transform = TrainValSplitTransform.from_config(val_config)
        val_data, _ = val_transform(data)

        # Verify they are complementary (no overlap, together they form the original)
        # Use the unique identifier column to check overlap since indices are reset
        train_ids = set(train_data["id"])
        val_ids = set(val_data["id"])
        original_ids = set(data["id"])

        assert len(train_ids.intersection(val_ids)) == 0  # No overlap
        assert train_ids.union(val_ids) == original_ids  # Together form original
        assert len(train_data) + len(val_data) == len(data)  # Sizes add up
