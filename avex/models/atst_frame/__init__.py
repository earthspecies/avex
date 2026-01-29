"""ATST Frame model implementation.

This module contains the Frame-level Audio Self-supervised Transformer (FrameATST)
implementation for audio representation learning.
"""

from .atst_encoder import Model
from .atst_frame import (
    FrameAST,
    FrameAST_base,
    FrameAST_large,
    FrameAST_small,
    FrameATST,
    FrameATSTLightningModule,
    get_scene_embedding,
    get_timestamp_embedding,
    load_model,
)

__all__ = [
    "Model",
    "FrameAST",
    "FrameATST",
    "FrameATSTLightningModule",
    "FrameAST_base",
    "FrameAST_large",
    "FrameAST_small",
    "get_scene_embedding",
    "get_timestamp_embedding",
    "load_model",
]
