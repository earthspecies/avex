"""
Probe utilities for working with probe configurations and probe classes.

This module maintains a static registry of probe classes. Users construct
:class:`ProbeConfig` objects directly or load them from YAML, and pass them
to the factory functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import yaml

from representation_learning.configs import ProbeConfig
from representation_learning.models.probes.attention_probe import AttentionProbe
from representation_learning.models.probes.linear_probe import LinearProbe
from representation_learning.models.probes.lstm_probe import LSTMProbe
from representation_learning.models.probes.mlp_probe import MLPProbe
from representation_learning.models.probes.transformer_probe import TransformerProbe

logger = logging.getLogger(__name__)

# Static probe registry: maps probe name to (class, default_config)
# default_config can be None or a ProbeConfig instance with default values
_PROBE_REGISTRY: Dict[str, Tuple[Type, Optional[ProbeConfig]]] = {
    "linear": (LinearProbe, None),
    "mlp": (MLPProbe, None),
    "lstm": (LSTMProbe, None),
    "attention": (AttentionProbe, None),
    "transformer": (TransformerProbe, None),
}

# Backward compatibility: map to just classes for get_probe_class()
_PROBE_CLASSES: Dict[str, Type] = {name: cls for name, (cls, _) in _PROBE_REGISTRY.items() if cls is not None}


def load_probe_config(yaml_path: str | Path) -> ProbeConfig:
    """Load a ProbeConfig directly from YAML.

    This function extracts the probe configuration from YAML files, supporting:
    1) Files with top-level probe fields directly at root level
    2) Files with probe_config: {...} key

    Args:
        yaml_path: Path to a YAML file (str or Path) containing a probe definition.

    Returns:
        A validated ProbeConfig instance

    Raises:
        ValueError: If YAML structure is invalid
    """
    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("YAML must define a mapping for probe specification")

    # Accept either top-level probe_config or direct ProbeConfig fields at root
    probe_dict = data.get("probe_config", data)
    if not isinstance(probe_dict, dict):
        raise ValueError("Invalid probe specification structure in YAML")

    return ProbeConfig(**probe_dict)


def get_probe_class(name: str) -> Optional[Type]:
    """Get a registered probe class by name.

    Args:
        name: Name of the probe class (e.g., "linear", "mlp", "attention")

    Returns:
        The registered probe class if found, None otherwise
    """
    name_lower = name.lower()
    if name_lower not in _PROBE_CLASSES:
        logger.info(f"Probe class '{name}' is not registered")
        return None
    return _PROBE_CLASSES[name_lower]


def list_probe_classes() -> list[str]:
    """Return a list of registered probe class names.

    Returns:
        List of registered probe class names
    """
    return list(_PROBE_CLASSES.keys())
