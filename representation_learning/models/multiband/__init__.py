"""Multiband audio processing for variable sample rate inputs.

This module provides tools for processing high sample rate audio by
splitting it into frequency bands via heterodyning, processing each
band through a shared backbone, and fusing the results.
"""

from representation_learning.models.multiband.fusion import (
    AttentionFusion,
    BaseFusion,
    ConcatFusion,
    GatedFusion,
    HybridGatedFusion,
    LogitFusion,
    build_fusion,
)
from representation_learning.models.multiband.model import MultibandModel
from representation_learning.models.multiband.wrapper import MultibandWrapper
from representation_learning.models.multiband.scoring import BandScorer, BandScores
from representation_learning.models.multiband.transforms import (
    HeterodyneCfg,
    HeterodyneToBaseband,
    MultibandTransform,
)

__all__ = [
    # Transforms
    "MultibandTransform",
    "HeterodyneToBaseband",
    "HeterodyneCfg",
    # Scoring
    "BandScorer",
    "BandScores",
    # Fusion
    "BaseFusion",
    "ConcatFusion",
    "AttentionFusion",
    "GatedFusion",
    "HybridGatedFusion",
    "LogitFusion",
    "build_fusion",
    # Model
    "MultibandModel",
    # Wrapper (use with any backbone)
    "MultibandWrapper",
]
