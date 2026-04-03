"""BEATs (Bidirectional Encoder representation from Audio Transformers) model."""

from .backbone import TransformerEncoder
from .beats import BEATs, BEATsConfig, official_beats_architecture_config

__all__ = [
    "BEATs",
    "BEATsConfig",
    "TransformerEncoder",
    "official_beats_architecture_config",
]
