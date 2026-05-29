"""Unit tests for FillLabelsFromAnswer transform and _is_empty_labels helper.

These tests require esp_data which is an internal dependency.
They are skipped when esp_data is not installed.
"""

import numpy as np
import pandas as pd
import pytest

# Skip entire module if esp_data is not installed (internal dependency)
# Must be before imports that trigger esp_data loading (e.g., avex.data.transforms)
pytest.importorskip("esp_data")

from avex.data.transforms import (  # noqa: E402
    FillLabelsFromAnswer,
    FillLabelsFromAnswerConfig,
    _is_empty_labels,
)


def _to_pandas(result: object) -> pd.DataFrame:
    """Extract a pandas DataFrame from a PandasBackend result.

    Returns
    -------
    pd.DataFrame
        The unwrapped pandas DataFrame.
    """
    frame = getattr(result, "unwrap", result)
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    return frame


class TestIsEmptyLabels:
    """Test cases for the _is_empty_labels helper."""

    def test_none_is_empty(self) -> None:
        assert _is_empty_labels(None) is True

    def test_nan_is_empty(self) -> None:
        assert _is_empty_labels(float("nan")) is True
        assert _is_empty_labels(np.nan) is True

    def test_empty_list_is_empty(self) -> None:
        assert _is_empty_labels([]) is True
        assert _is_empty_labels([None, None]) is True
        assert _is_empty_labels(np.array([])) is True

    def test_legit_float_is_not_empty(self) -> None:
        """A real float label must not be treated as empty NaN."""
        assert _is_empty_labels(3.0) is False

    def test_populated_list_is_not_empty(self) -> None:
        assert _is_empty_labels(["a"]) is False
        assert _is_empty_labels(["a", None]) is False


class TestFillLabelsFromAnswer:
    """Test cases for FillLabelsFromAnswer on pandas input."""

    def _run(self, df: pd.DataFrame, **kwargs: object) -> tuple[pd.DataFrame, dict]:
        config = FillLabelsFromAnswerConfig(type="fill_labels_from_answer", **kwargs)
        transform = FillLabelsFromAnswer.from_config(config)
        result, metadata = transform(df)
        return _to_pandas(result), metadata

    def test_fills_empty_from_answer(self) -> None:
        """Empty list / NaN targets are filled from the answer column."""
        df = pd.DataFrame(
            {
                "labels_as_list": [[], [None], np.nan],
                "answer": ["crow", "robin", "finch"],
            }
        )
        out, metadata = self._run(df)

        assert out["labels_as_list"].tolist() == [["crow"], ["robin"], ["finch"]]
        assert metadata["fill_labels_from_answer"]["filled"] == 3

    def test_comma_separated_answer_splits(self) -> None:
        """Comma-separated answers become multiple labels."""
        df = pd.DataFrame({"labels_as_list": [[]], "answer": ["crow, robin , finch"]})
        out, _ = self._run(df)
        assert out["labels_as_list"].tolist() == [["crow", "robin", "finch"]]

    def test_ignore_answers_not_used(self) -> None:
        """Answers in ignore_answers leave the target empty."""
        df = pd.DataFrame({"labels_as_list": [[], []], "answer": ["None", "crow"]})
        out, metadata = self._run(df)
        assert out["labels_as_list"].tolist() == [[], ["crow"]]
        assert metadata["fill_labels_from_answer"]["filled"] == 1

    def test_populated_target_untouched(self) -> None:
        """Rows with existing labels are left as-is (no-op for val/test)."""
        df = pd.DataFrame({"labels_as_list": [["owl"]], "answer": ["crow"]})
        out, metadata = self._run(df)
        assert out["labels_as_list"].tolist() == [["owl"]]
        assert metadata["fill_labels_from_answer"]["filled"] == 0

    def test_filled_count_accurate(self) -> None:
        """The filled counter matches the number of actually-filled rows."""
        df = pd.DataFrame(
            {
                "labels_as_list": [[], ["owl"], np.nan, []],
                "answer": ["crow", "robin", "None", "finch"],
            }
        )
        _, metadata = self._run(df)
        # rows 0 and 3 filled; row 1 already populated; row 2 answer ignored
        assert metadata["fill_labels_from_answer"]["filled"] == 2

    def test_missing_columns_raise(self) -> None:
        """Missing source/target columns raise ValueError."""
        with pytest.raises(ValueError, match="Target column"):
            self._run(pd.DataFrame({"answer": ["crow"]}))
        with pytest.raises(ValueError, match="Source column"):
            self._run(pd.DataFrame({"labels_as_list": [[]]}))
