"""Tests for _embedding_cache_matches cache-validation logic.

Covers the branches that the clustering/retrieval reuse paths key off:
missing file, incomplete extraction, aggregation mismatch (including the
legacy `aggregation` attr fallback), and the matching happy path.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from avex.utils.utils import _embedding_cache_matches


def _write_cache(
    path: Path,
    *,
    aggregation: str | None = "mean",
    legacy_aggregation: str | None = None,
    extraction_complete: bool | None = True,
) -> None:
    """Write a minimal HDF5 cache file with the given metadata attrs."""
    with h5py.File(str(path), "w") as h5f:
        h5f.create_dataset("labels", data=[0, 1, 0])
        if aggregation is not None:
            h5f.attrs["embedding_aggregation"] = aggregation
        if legacy_aggregation is not None:
            h5f.attrs["aggregation"] = legacy_aggregation
        if extraction_complete is not None:
            h5f.attrs["extraction_complete"] = extraction_complete


def test_missing_file_returns_false(tmp_path: Path) -> None:
    assert _embedding_cache_matches(tmp_path / "absent.h5", expected_aggregation="mean") is False


def test_matching_aggregation_and_complete_returns_true(tmp_path: Path) -> None:
    path = tmp_path / "c.h5"
    _write_cache(path, aggregation="mean", extraction_complete=True)
    assert _embedding_cache_matches(path, expected_aggregation="mean") is True


def test_aggregation_mismatch_returns_false(tmp_path: Path) -> None:
    """A clustering cache stored as 'mean' must not match probe agg 'none'."""
    path = tmp_path / "c.h5"
    _write_cache(path, aggregation="mean", extraction_complete=True)
    assert _embedding_cache_matches(path, expected_aggregation="none") is False


def test_incomplete_extraction_returns_false(tmp_path: Path) -> None:
    path = tmp_path / "c.h5"
    _write_cache(path, aggregation="mean", extraction_complete=False)
    assert _embedding_cache_matches(path, expected_aggregation="mean") is False


def test_missing_extraction_complete_attr_returns_false(tmp_path: Path) -> None:
    """Pre-existing caches without extraction_complete are invalidated once."""
    path = tmp_path / "c.h5"
    _write_cache(path, aggregation="mean", extraction_complete=None)
    assert _embedding_cache_matches(path, expected_aggregation="mean") is False


def test_legacy_aggregation_attr_fallback(tmp_path: Path) -> None:
    """Falls back to the legacy `aggregation` attr when `embedding_aggregation` absent."""
    path = tmp_path / "c.h5"
    _write_cache(path, aggregation=None, legacy_aggregation="max", extraction_complete=True)
    assert _embedding_cache_matches(path, expected_aggregation="max") is True
    assert _embedding_cache_matches(path, expected_aggregation="mean") is False


@pytest.mark.parametrize("agg", ["mean", "max", "cls_token"])
def test_pooled_aggregations_roundtrip(tmp_path: Path, agg: str) -> None:
    path = tmp_path / f"c_{agg}.h5"
    _write_cache(path, aggregation=agg, extraction_complete=True)
    assert _embedding_cache_matches(path, expected_aggregation=agg) is True
