"""API module for representation learning.

This module provides the main public API for the representation learning framework,
including model loading, probe management, and utility functions.
"""

from __future__ import annotations

# Probe API - import probe utilities
from representation_learning.models.probes import (
    build_probe_from_config,
    get_probe,
    load_probe_config_from_yaml,
)

# Model API - import what's actually exported from models.utils
from representation_learning.models.utils import (
    describe_model,
    get_model_spec,
    list_models,
    load_model,
    register_model,
)

__all__ = [
    # Model API
    "load_model",
    "list_models",
    "describe_model",
    "get_model_spec",
    "register_model",
    # Probe API
    "get_probe",
    "build_probe_from_config",
    "load_probe_config_from_yaml",
]
