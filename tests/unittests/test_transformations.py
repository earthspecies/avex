"""
Unit tests for data transformations.
"""

import pandas as pd
import pytest

from esp_data_temp.transforms import (
    Filter,
    FilterConfig,
    Subsample,
    SubsampleConfig,
    transform_from_config,
)

# TODO (milad) add tests for returned metadata


def test_filter_dataframe() -> None:
    """Test filtering a pandas DataFrame."""
    # Create test data
    df = pd.DataFrame(
        {
            "source": ["xeno-canto", "iNaturalist", "Watkins", "other"],
            "class": ["birds", "mammals", "amphibians", "reptiles"],
            "value": [1, 2, 3, 4],
        }
    )

    # Test include operation
    config = FilterConfig(
        type="filter",
        property="source",
        values=["xeno-canto", "iNaturalist"],
        mode="include",
    )
    filter_transform = Filter.from_config(config)
    filtered_df, _ = filter_transform(df)

    assert len(filtered_df) == 2
    assert set(filtered_df["source"]) == {"xeno-canto", "iNaturalist"}

    # Test exclude operation
    config = FilterConfig(
        type="filter",
        property="source",
        values=["xeno-canto", "iNaturalist"],
        mode="exclude",
    )
    filter_transform = Filter.from_config(config)
    filtered_df, _ = filter_transform(df)

    assert len(filtered_df) == 2
    assert set(filtered_df["source"]) == {"Watkins", "other"}


def test_subsample_dataframe() -> None:
    """Test subsampling a pandas DataFrame."""
    # Create test data with known class distribution
    df = pd.DataFrame(
        {
            "class": ["birds"] * 100 + ["mammals"] * 100 + ["amphibians"] * 100,
            "value": range(300),
        }
    )

    # Test subsampling with different ratios
    config = SubsampleConfig(
        type="subsample",
        property="class",
        ratios={"birds": 0.5, "mammals": 0.3, "amphibians": 0.7},
    )
    subsample_transform = Subsample.from_config(config)
    subsampled_df, _ = subsample_transform(df)

    # Check that the ratios are approximately correct
    class_counts = subsampled_df["class"].value_counts()
    assert abs(class_counts["birds"] / 100 - 0.5) < 0.1
    assert abs(class_counts["mammals"] / 100 - 0.3) < 0.1
    assert abs(class_counts["amphibians"] / 100 - 0.7) < 0.1

    # Test with 'other' class
    config = SubsampleConfig(
        type="subsample",
        property="class",
        ratios={"birds": 0.5, "other": 0.2},
    )
    subsample_transform = Subsample.from_config(config)
    subsampled_df, _ = subsample_transform(df)

    # Check that 'other' class (mammals + amphibians) is subsampled correctly
    other_count = len(
        subsampled_df[subsampled_df["class"].isin(["mammals", "amphibians"])]
    )
    assert abs(other_count / 200 - 0.2) < 0.1


def test_transform_from_config() -> None:
    """Test building transformations from configuration."""
    # Test building a single filter transform
    configs = [
        FilterConfig(
            type="filter",
            property="source",
            values=["xeno-canto", "iNaturalist"],
            mode="include",
        )
    ]
    transforms = [transform_from_config(c) for c in configs]
    assert isinstance(transforms[0], Filter)

    # Test building a single subsample transform
    configs = [
        SubsampleConfig(
            type="subsample",
            property="class",
            ratios={"birds": 0.5},
        )
    ]
    transforms = [transform_from_config(c) for c in configs]
    assert isinstance(transforms[0], Subsample)

    # Test building multiple transforms
    configs = [
        FilterConfig(
            type="filter",
            property="source",
            values=["xeno-canto"],
            mode="include",
        ),
        SubsampleConfig(
            type="subsample",
            property="class",
            ratios={"birds": 0.5},
        ),
    ]
    transforms = [transform_from_config(c) for c in configs]
    assert isinstance(transforms[0], Filter)
    assert isinstance(transforms[1], Subsample)

    # Test invalid transform type
    configs = {"invalid": {}}
    with pytest.raises(ValueError):
        build_transforms(configs)
