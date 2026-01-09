import pandas as pd
import pytest
from esp_data.backends import PandasBackend
from esp_data.transforms import LabelFromFeature


@pytest.mark.parametrize(
    "feature, df, expected_labels, expected_map",
    [
        # Single column, all strings
        (
            "col1",
            pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]}),
            [1, 0, 1, 2],
            {"apple": 0, "banana": 1, "orange": 2},
        ),
        # Single column, with NaN
        (
            "col1",
            pd.DataFrame({"col1": ["banana", "apple", float("nan"), "orange"]}),
            [1, 0, 2],
            {"apple": 0, "banana": 1, "orange": 2},
        ),
    ],
)
def test_label_from_feature(
    feature: str,
    df: pd.DataFrame,
    expected_labels: list[int],
    expected_map: dict[str, int],
) -> None:
    t = LabelFromFeature(feature=feature)
    backend = PandasBackend(df.copy())
    backend_out, meta = t(backend)
    df_out = backend_out.unwrap
    assert df_out["label"].tolist() == expected_labels
    assert meta["label_map"] == expected_map
    assert meta["num_classes"] == len(expected_map)


def test_label_from_feature_with_label_map() -> None:
    df = pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
    label_map = {"apple": 0, "banana": 1, "orange": 2, "grape": 3}
    t = LabelFromFeature(feature="col1", label_map=label_map)
    backend = PandasBackend(df.copy())
    backend_out, meta = t(backend)
    df_out = backend_out.unwrap
    assert df_out["label"].tolist() == [1, 0, 1, 2]
    assert meta["label_map"] == label_map
    assert meta["num_classes"] == len(label_map)


def test_label_from_feature_with_noncontiguous_indices() -> None:
    df = pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
    label_map = {"apple": 100, "banana": 101, "orange": 102}
    t = LabelFromFeature(feature="col1", label_map=label_map)
    backend = PandasBackend(df.copy())
    backend_out, meta = t(backend)
    df_out = backend_out.unwrap
    assert df_out["label"].tolist() == [101, 100, 101, 102]
    assert meta["label_map"] == label_map
    assert meta["num_classes"] == len(label_map)


def test_label_from_feature_label_map_remains_none() -> None:
    df = pd.DataFrame({"col1": ["banana", "apple", "banana", "orange"]})
    t = LabelFromFeature(feature="col1")
    assert t.label_map is None
    backend = PandasBackend(df.copy())
    backend_out, meta = t(backend)
    df_out = backend_out.unwrap
    assert t.label_map is None
    assert sorted(meta["label_map"].keys()) == ["apple", "banana", "orange"]
    assert meta["num_classes"] == 3
    assert df_out["label"].tolist() == [1, 0, 1, 2]
