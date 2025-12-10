"""BEATs (Bidirectional Encoder representation from Audio Transformers) model.

Supports both original BEATs and OpenBEATs checkpoints with base and large model sizes.
"""

from .backbone import (
    MultiheadAttention,
    TransformerEncoder,
    TransformerSentenceEncoderLayer,
)
from .beats import (
    BEATS_BASE_CONFIG,
    BEATS_LARGE_CONFIG,
    BEATS_OUTPUT_DIMS,
    BEATS_SIZE_CONFIGS,
    BEATs,
    BEATsConfig,
)
from .modules import (
    GLU_Linear,
    GradMultiply,
    SamePad,
    Swish,
    get_activation_fn,
    quant_noise,
)

__all__ = [
    "BEATs",
    "BEATsConfig",
    "BEATS_BASE_CONFIG",
    "BEATS_LARGE_CONFIG",
    "BEATS_SIZE_CONFIGS",
    "BEATS_OUTPUT_DIMS",
    "MultiheadAttention",
    "TransformerEncoder",
    "TransformerSentenceEncoderLayer",
    "GLU_Linear",
    "GradMultiply",
    "SamePad",
    "Swish",
    "get_activation_fn",
    "quant_noise",
]
