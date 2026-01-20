"""
Probe utilities for the representation learning framework.

This module provides utilities for managing and creating probes,
mirroring the model utilities structure.

Key components:
- Registry: Discover probe classes and load ProbeConfig from YAML
- Factory: Build probe instances from configurations
"""

from __future__ import annotations

from .factory import build_probe_from_config
from .registry import load_probe_config

__all__ = [
    # Factory functions
    "build_probe_from_config",
    # Config helpers
    "load_probe_config",
]
