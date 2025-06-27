import pandas as pd
from esp_data.transforms import Subsample, SubsampleConfig

# TODO (milad) add tests for returned metadata


def test_subsample() -> None:
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


def test_subsample_manual_vs_config() -> None:
    """Test that manual instantiation and from_config produce the same result."""
    df = pd.DataFrame(
        {
            "class": ["birds"] * 100 + ["mammals"] * 100 + ["amphibians"] * 100,
            "value": range(300),
        }
    )
    ratios = {"birds": 0.5, "mammals": 0.3, "amphibians": 0.7}
    # Manual instantiation
    manual_transform = Subsample(property="class", ratios=ratios)
    manual_result, _ = manual_transform(df)

    # Using config and from_config
    config = SubsampleConfig(type="subsample", property="class", ratios=ratios)
    config_transform = Subsample.from_config(config)
    config_result, _ = config_transform(df)

    # Sort and reset index for comparison
    manual_sorted = manual_result.sort_values(by=["class", "value"]).reset_index(
        drop=True
    )
    config_sorted = config_result.sort_values(by=["class", "value"]).reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(manual_sorted, config_sorted)
