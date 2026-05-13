"""Shared PyTorch numerics profile helpers for regression fixtures.

Official model output fingerprints and evaluate probe baselines are keyed by
the installed PyTorch release band, not the Python interpreter version.
"""

from __future__ import annotations

from typing import Final

import torch

TORCH_FINGERPRINT_PROFILES: Final[tuple[str, ...]] = ("torch_2_5_0", "torch_2_11_0")


def torch_release_tuple(torch_version: str | None = None) -> tuple[int, int, int]:
    """Parse a PyTorch version string into a comparable release triple.

    Args:
        torch_version: Optional version string. Defaults to ``torch.__version__``.

    Returns:
        ``(major, minor, patch)`` with missing segments treated as zero.
    """
    raw = (torch_version or torch.__version__).split("+", 1)[0]
    parts = raw.split(".")
    nums: list[int] = []
    for part in parts[:3]:
        nums.append(int(part))
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def torch_fingerprint_profile(torch_version: str | None = None) -> str:
    """Return the fingerprint profile for the active PyTorch release band.

    Args:
        torch_version: Optional version string. Defaults to ``torch.__version__``.

    Returns:
        ``torch_2_11_0`` when the release is at least 2.11.0, otherwise
        ``torch_2_5_0``.
    """
    release = torch_release_tuple(torch_version)
    if release >= (2, 11, 0):
        return "torch_2_11_0"
    return "torch_2_5_0"
