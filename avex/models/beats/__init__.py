"""BEATs (Bidirectional Encoder representation from Audio Transformers) model."""

from .backbone import TransformerEncoder
from .beats import BEATs, BEATsConfig

__all__ = [
    "BEATs",
    "BEATsConfig",
    "TransformerEncoder",
]
