"""Tests for embedding cache metadata.

These tests require esp_data which is an internal dependency.
They are skipped when esp_data is not installed.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest
import torch

# Skip entire module if esp_data is not installed (internal dependency)
# Must be before imports that trigger esp_data loading
pytest.importorskip("esp_data")

from avex.evaluation.embedding_utils import save_embeddings_arrays  # noqa: E402


def test_save_embeddings_arrays_records_aggregation_metadata(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.h5"
    embeddings = {"layer": torch.randn(3, 8)}
    labels = torch.tensor([0, 1, 0])

    save_embeddings_arrays(
        embeddings,
        labels,
        path,
        num_labels=2,
        aggregation="mean",
    )

    with h5py.File(path, "r") as h5f:
        assert h5f.attrs["embedding_aggregation"] == "mean"
        assert h5f.attrs["aggregation"] == "mean"
        assert list(h5f.attrs["stored_embedding_rank"]) == [1]
        assert list(h5f.attrs["layer_names"]) == ["layer"]
        assert list(h5f.attrs["embedding_dims"]) == ["(8,)"]
