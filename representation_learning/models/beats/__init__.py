"""BEATs (Bidirectional Encoder representation from Audio Transformers) model."""

from .backbone import (
    MultiheadAttention,
    TransformerEncoder,
    TransformerSentenceEncoderLayer,
)
from .beats import BEATs, BEATsConfig
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
