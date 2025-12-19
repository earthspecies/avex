"""
Probe utilities for the representation learning framework.

This module provides utilities for managing and creating probes,
mirroring the model utilities structure.

Key components:
- Registry: Discover probe classes and load ProbeConfig from YAML
- Factory: Build probe instances from configurations
"""

from __future__ import annotations

from .factory import (
    build_probe_from_config_offline,
    build_probe_from_config_online,
)
from .registry import load_probe_config_from_yaml

__all__ = [
    # Factory functions
    "build_probe_from_config_online",
    "build_probe_from_config_offline",
    # Config helpers
    "load_probe_config_from_yaml",
]
