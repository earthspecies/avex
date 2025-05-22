import pandas as pd
import pytest

from esp_data_temp.transforms import MultiLabelFromFeatures


@pytest.mark.parametrize(
    "features, data, expected_labels, expected_map",
    [
        # Single column, all strings
        (
            ["col1"],
            pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]}),
            [[1], [0], [1], [2]],
            {"apple": 0, "banana": 1, "orange": 2},
        ),
        # Single column, all lists
        (
            ["col1"],
            pd.DataFrame(
                {
                    "col1": [
                        ["banana", "apple"],
                        ["apple"],
                        ["banana", "orange"],
                        ["orange"],
                    ]
                }
            ),
            [[0, 1], [0], [1, 2], [2]],
            {"apple": 0, "banana": 1, "orange": 2},
        ),
        # Single column, mix of strings and lists
        (
            ["col1"],
            pd.DataFrame(
                {"col1": ["banana", ["apple"], ["banana", "orange"], "orange"]}
            ),
            [[1], [0], [1, 2], [2]],
            {"apple": 0, "banana": 1, "orange": 2},
        ),
        # Multiple columns, mix of strings and lists
        (
            ["col1", "col2"],
            pd.DataFrame(
                {
                    "col1": ["banana", ["apple"], ["banana", "orange"], "orange"],
                    "col2": [["grape"], "kiwi", ["grape", "melon"], "melon"],
                }
            ),
            [[1, 2], [0, 3], [1, 2, 4, 5], [4, 5]],
            {"apple": 0, "banana": 1, "grape": 2, "kiwi": 3, "melon": 4, "orange": 5},
        ),
    ],
)
def test_multilabel_from_features(
    features: list[str],
    data: pd.DataFrame,
    expected_labels: list[list[str]],
    expected_map: dict[str, int],
) -> None:
    t = MultiLabelFromFeatures(features=features)
    df_out, meta = t(data.copy())
    assert df_out["label"].tolist() == expected_labels
    assert meta["label_map"] == expected_map
    assert meta["num_classes"] == len(expected_map)
