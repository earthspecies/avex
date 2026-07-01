"""Unit tests for RLSubsampleTransform.

Datasets are always loaded through an esp_data backend (pandas or polars), so
these tests exercise the transform against real backend objects.

These tests require esp_data which is an internal dependency.
They are skipped when esp_data is not installed.
"""

import pandas as pd
import pytest

# Skip entire module if esp_data is not installed (internal dependency)
# Must be before imports that trigger esp_data loading (e.g., avex.data.transforms)
pytest.importorskip("esp_data")

from esp_data.backends import PandasBackend, PolarsBackend  # noqa: E402

from avex.data.transforms import (  # noqa: E402
    RLSubsampleConfig,
    RLSubsampleTransform,
)


def _pandas_backend(n: int) -> PandasBackend:
    return PandasBackend(
        pd.DataFrame(
            {
                "path": [f"data/sample_{i}.wav" for i in range(n)],
                "label": [str(i % 5) for i in range(n)],
            }
        )
    )


def _polars_backend(n: int) -> PolarsBackend:
    pl = pytest.importorskip("polars")
    return PolarsBackend(
        pl.DataFrame(
            {
                "path": [f"data/sample_{i}.wav" for i in range(n)],
                "label": [str(i % 5) for i in range(n)],
            }
        )
    )


class TestRLSubsampleTransform:
    """Test cases for RLSubsampleTransform on esp_data backends."""

    @pytest.mark.parametrize("backend_cls", [PandasBackend, PolarsBackend])
    def test_ratio_subsample(self, backend_cls: type) -> None:
        """Both pandas and polars backends are sampled via sample_rows()."""
        data = _pandas_backend(100) if backend_cls is PandasBackend else _polars_backend(100)
        config = RLSubsampleConfig(type="rl_subsample", ratio=0.5, random_state=42)

        result, metadata = RLSubsampleTransform.from_config(config)(data)

        assert isinstance(result, backend_cls)
        assert len(result) == 50
        assert metadata["rl_subsample"]["original_size"] == 100
        assert metadata["rl_subsample"]["sampled_size"] == 50
        assert metadata["rl_subsample"]["ratio"] == 0.5

    def test_max_samples_cap(self) -> None:
        """max_samples caps the sampled count below ratio."""
        data = _pandas_backend(100)
        config = RLSubsampleConfig(type="rl_subsample", ratio=0.9, max_samples=10, random_state=42)

        result, metadata = RLSubsampleTransform.from_config(config)(data)

        assert len(result) == 10
        assert metadata["rl_subsample"]["max_samples"] == 10

    @pytest.mark.parametrize("backend_cls", [PandasBackend, PolarsBackend])
    def test_zero_ratio_returns_empty(self, backend_cls: type) -> None:
        """ratio=0 yields an empty backend (slice [0:0]), not a crash."""
        data = _pandas_backend(100) if backend_cls is PandasBackend else _polars_backend(100)
        config = RLSubsampleConfig(type="rl_subsample", ratio=0.0, random_state=42)

        result, metadata = RLSubsampleTransform.from_config(config)(data)

        assert isinstance(result, backend_cls)
        assert len(result) == 0
        assert metadata["rl_subsample"]["sampled_size"] == 0

    def test_empty_data(self) -> None:
        """An empty input backend returns empty with zeroed metadata."""
        data = _pandas_backend(0)
        config = RLSubsampleConfig(type="rl_subsample", ratio=0.5)

        result, metadata = RLSubsampleTransform.from_config(config)(data)

        assert len(result) == 0
        assert metadata["rl_subsample"]["original_size"] == 0
        assert metadata["rl_subsample"]["sampled_size"] == 0

    def test_reproducibility(self) -> None:
        """Same random_state produces identical samples."""
        data = _pandas_backend(100)
        config = RLSubsampleConfig(type="rl_subsample", ratio=0.3, random_state=123)

        result1, _ = RLSubsampleTransform.from_config(config)(data)
        result2, _ = RLSubsampleTransform.from_config(config)(data)

        pd.testing.assert_frame_equal(result1.unwrap, result2.unwrap)

    def test_invalid_ratio(self) -> None:
        """ratio outside [0, 1] is rejected."""
        with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
            RLSubsampleTransform(ratio=1.5)

    def test_invalid_max_samples(self) -> None:
        """max_samples < 1 is rejected."""
        with pytest.raises(ValueError, match="max_samples must be >= 1"):
            RLSubsampleTransform(max_samples=0)
