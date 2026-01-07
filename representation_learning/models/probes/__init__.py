"""Probe models for flexible representation learning evaluation."""

from representation_learning.models.probes.attention_probe import (
    AttentionProbe,
)
from representation_learning.models.probes.get_probe import get_probe
from representation_learning.models.probes.linear_probe import LinearProbe
from representation_learning.models.probes.lstm_probe import LSTMProbe
from representation_learning.models.probes.mlp_probe import MLPProbe
from representation_learning.models.probes.transformer_probe import (
    TransformerProbe,
)

# Import probe utilities
from representation_learning.models.probes.utils import (
    build_probe_from_config,
    load_probe_config,
)

__all__ = [
    # Legacy factory (public API)
    "get_probe",
    # Utility functions
    "build_probe_from_config",
    "load_probe_config",
    "AttentionProbe",
    "LinearProbe",
    "LSTMProbe",
    "MLPProbe",
    "TransformerProbe",
]
