from typing import Callable, Literal

import pandas as pd

from esp_data_temp.transforms import Filter, FilterConfig, transform_from_config

# TODO (milad) add tests for returned metadata


def test_filter() -> None:
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


def test_filter_transform_methods_equivalence() -> None:
    """
    Test that transform_from_config, manual, and from_config produce the same result for
    Filter (include/exclude).
    """

    def _get_sorted_result(df: pd.DataFrame, transform: Callable) -> pd.DataFrame:
        result, _ = transform(df)
        return result.sort_values(by=["source", "value"]).reset_index(drop=True)

    def _all_filter_methods(
        df: pd.DataFrame,
        property: str,
        values: list[str],
        mode: Literal["include", "exclude"],
    ) -> list[pd.DataFrame]:
        manual = Filter(property=property, values=values, mode=mode)
        config = FilterConfig(
            type="filter", property=property, values=values, mode=mode
        )
        from_config_transform = Filter.from_config(config)
        from_registry_transform = transform_from_config(config)
        return [
            _get_sorted_result(df, manual),
            _get_sorted_result(df, from_config_transform),
            _get_sorted_result(df, from_registry_transform),
        ]

    df = pd.DataFrame(
        {
            "source": ["xeno-canto", "iNaturalist", "Watkins", "other"],
            "class": ["birds", "mammals", "amphibians", "reptiles"],
            "value": [1, 2, 3, 4],
        }
    )
    values = ["xeno-canto", "iNaturalist"]
    for mode in ["include", "exclude"]:
        results = _all_filter_methods(df, property="source", values=values, mode=mode)
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                pd.testing.assert_frame_equal(results[i], results[j])
