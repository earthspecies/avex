"""Tests for checking basic linear transformation.

Header info specific to ESP
"""

import torch
import torch.nn


def test_audiomae(device: str) -> None:
    """Test our linear implementation against the torch basic
    one.
    """

    from representation_learning.models.audiomae import Model
