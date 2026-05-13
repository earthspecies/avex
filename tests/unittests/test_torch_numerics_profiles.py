"""Unit tests for PyTorch numerics profile selection."""

from __future__ import annotations

import pytest

from tests.integration.torch_numerics_profiles import (
    TORCH_FINGERPRINT_PROFILES,
    torch_fingerprint_profile,
    torch_release_tuple,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.5.0", "torch_2_5_0"),
        ("2.5.0+cpu", "torch_2_5_0"),
        ("2.6.0", "torch_2_6_0"),
        ("2.7.1", "torch_2_6_0"),
        ("2.9.9", "torch_2_6_0"),
        ("2.10.0", "torch_2_11_0"),
        ("2.10.9", "torch_2_11_0"),
        ("2.11.0", "torch_2_11_0"),
        ("2.11.0+cu130", "torch_2_11_0"),
        ("2.12.1", "torch_2_11_0"),
    ],
)
def test_torch_fingerprint_profile_maps_release_bands(version: str, expected: str) -> None:
    """Profiles follow PyTorch release bands, not Python interpreter versions."""
    assert torch_fingerprint_profile(version) == expected
    assert expected in TORCH_FINGERPRINT_PROFILES


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.11.0+cu130", (2, 11, 0)),
        ("2.5.0", (2, 5, 0)),
        ("2.5", (2, 5, 0)),
    ],
)
def test_torch_release_tuple_strips_build_metadata(version: str, expected: tuple[int, int, int]) -> None:
    """Release parsing ignores CUDA or other local build suffixes."""
    assert torch_release_tuple(version) == expected
