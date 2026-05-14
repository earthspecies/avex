"""Tests for published safetensors artifact validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from avex.utils.safetensors_validation import (
    MIN_PUBLISHED_SAFETENSORS_BYTES,
    SafetensorsWeightsError,
    assert_safetensors_has_weights,
)


def test_assert_safetensors_has_weights_accepts_nonempty_artifact(tmp_path: Path) -> None:
    """A non-empty tensor map should pass publish validation."""
    artifact_path = tmp_path / "weights.safetensors"
    save_file({"weight": torch.zeros(32, 32)}, artifact_path)

    size_bytes = assert_safetensors_has_weights(artifact_path)

    assert size_bytes >= MIN_PUBLISHED_SAFETENSORS_BYTES


def test_assert_safetensors_has_weights_rejects_metadata_only_artifact(tmp_path: Path) -> None:
    """Metadata-only safetensors shells should fail publish validation."""
    artifact_path = tmp_path / "empty.safetensors"
    save_file({}, artifact_path, metadata={"format": "safetensors"})

    with pytest.raises(SafetensorsWeightsError, match="too small"):
        assert_safetensors_has_weights(artifact_path)
