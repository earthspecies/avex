"""OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder.

This module provides the OpenBEATs model implementation for audio representation
learning tasks, based on the BEATs architecture with enhancements for pre-training
and flash attention support.

Based on:
- Paper: https://arxiv.org/abs/2507.14129 (OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder)
- Original BEATs: https://arxiv.org/abs/2212.09058
- HuggingFace Collection: https://huggingface.co/collections/shikhar7ssu/openbeats
"""

from representation_learning.models.openbeats.openbeats import (
    OPENBEATS_BASE_CONFIG,
    OPENBEATS_GIANT_CONFIG,
    OPENBEATS_LARGE_CONFIG,
    OPENBEATS_TITAN_CONFIG,
    OpenBEATs,
    OpenBEATsConfig,
)

__all__ = [
    "OpenBEATs",
    "OpenBEATsConfig",
    "OPENBEATS_BASE_CONFIG",
    "OPENBEATS_LARGE_CONFIG",
    "OPENBEATS_GIANT_CONFIG",
    "OPENBEATS_TITAN_CONFIG",
]
