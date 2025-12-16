"""
Probe utilities for working with probe configurations and probe classes.

This module no longer maintains a global registry of named probe configs.
Instead, users are expected to construct :class:`ProbeConfig` objects
directly or load them from YAML, and pass them to the factory functions.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, Optional, Type

from representation_learning.configs import ProbeConfig

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a standard dep in this repo
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)

# Private global probe class registry for dynamic probe registration
_PROBE_CLASSES: Dict[str, Type] = {}


def _discover_probe_classes() -> Dict[str, Type]:
    """Discover all _BaseProbe subclasses in the probes package.

    Dynamically scans the probes package directory to find all probe classes,
    avoiding the need to manually list each probe.

    Returns:
        Dictionary mapping probe names to their classes
    """
    discovered = {}

    # Import _BaseProbe here to avoid circular imports
    try:
        from representation_learning.models.probes.base_probes import _BaseProbe
    except ImportError:
        logger.warning("Could not import _BaseProbe")
        return discovered

    # Mapping from module path to probe name(s)
    PROBE_NAME_MAPPING: Dict[str, list[str]] = {
        "representation_learning.models.probes.linear_probe": ["linear"],
        "representation_learning.models.probes.mlp_probe": ["mlp"],
        "representation_learning.models.probes.lstm_probe": ["lstm"],
        "representation_learning.models.probes.attention_probe": ["attention"],
        "representation_learning.models.probes.transformer_probe": ["transformer"],
    }

    # Get the probes package path
    try:
        probes_pkg = importlib.import_module("representation_learning.models.probes")
        probes_path = Path(probes_pkg.__file__).parent if probes_pkg.__file__ else None
    except ImportError:
        logger.warning("Could not import representation_learning.models.probes package")
        return discovered

    if not probes_path or not probes_path.exists():
        logger.warning(f"Probes package path not found: {probes_path}")
        return discovered

    # Scan Python files in the probes directory
    for py_file in probes_path.glob("*_probe.py"):
        # Skip base_probes.py and get_probe.py
        if py_file.name in ("base_probes.py", "get_probe.py"):
            continue

        # Convert file path to module name
        try:
            module_name = f"representation_learning.models.probes.{py_file.stem}"
        except ValueError:
            continue

        try:
            module = importlib.import_module(module_name)

            # Find all _BaseProbe subclasses in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Skip if not a _BaseProbe subclass, or if it's _BaseProbe itself
                if not issubclass(obj, _BaseProbe) or obj is _BaseProbe:
                    continue

                # Skip if it's imported from another module (not defined here)
                if obj.__module__ != module_name:
                    continue

                # Get probe name(s) for this module
                probe_names = PROBE_NAME_MAPPING.get(module_name, [])

                if not probe_names:
                    # Infer probe name from class name
                    if name.endswith("Probe"):
                        probe_name = name[:-5].lower()  # Remove "Probe" suffix
                        probe_names = [probe_name]
                    else:
                        probe_names = [name.lower()]

                # Register all probe names for this class
                for probe_name in probe_names:
                    discovered[probe_name] = obj
                    logger.debug(f"Discovered probe class: {probe_name} -> {obj.__name__} from {module_name}")

        except ImportError as e:
            # Some modules may have optional dependencies
            logger.debug(f"Could not import {module_name}: {e}")
        except Exception as e:
            logger.warning(f"Error discovering probes in {module_name}: {e}")

    return discovered


def _auto_register_probe_classes() -> None:
    """Automatically register built-in probe classes from the probes/ directory.

    This registers all standard probe classes so they can be used via the registry
    without manual registration. Probe classes are discovered dynamically by scanning
    the probes package for ProbeBase subclasses.
    """
    try:
        discovered = _discover_probe_classes()

        for probe_name, probe_class in discovered.items():
            _PROBE_CLASSES[probe_name] = probe_class

        logger.info(f"Auto-registered {len(discovered)} probe classes: {sorted(discovered.keys())}")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to auto-register probe classes: {e}")


def load_probe_config_from_yaml(yaml_path: str | Path) -> ProbeConfig:
    """Load a ProbeConfig directly from YAML.

    This function extracts the probe configuration from YAML files, supporting:
    1) Files with top-level probe fields directly at root level
    2) Files with probe_config: {...} key

    Args:
        yaml_path: Path to a YAML file (str or Path) containing a probe definition.

    Returns:
        A validated ProbeConfig instance

    Raises:
        ValueError: If YAML cannot be parsed into a ProbeConfig
    """
    if yaml is None:
        raise ValueError("PyYAML not available to parse YAML files")

    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("YAML must define a mapping for probe specification")

    # Accept either top-level probe_config or direct ProbeConfig fields at root
    probe_dict = data.get("probe_config", data)
    if not isinstance(probe_dict, dict):
        raise ValueError("Invalid probe specification structure in YAML")

    try:
        return ProbeConfig(**probe_dict)
    except Exception as e:
        raise ValueError(f"Failed to build ProbeConfig from YAML: {e}") from e


def get_probe_class(name: str) -> Optional[Type]:
    """Get a registered probe class by name.

    Args:
        name: Name of the probe class

    Returns:
        The registered probe class if found, None otherwise
    """
    if name not in _PROBE_CLASSES:
        logger.info(f"Probe class '{name}' is not registered")
        return None
    return _PROBE_CLASSES[name]


def list_probe_classes() -> list[str]:
    """Return a list of registered probe class names.

    Returns:
        List of registered probe class names
    """
    return list(_PROBE_CLASSES.keys())


# Initialize probe class registry at module import time
_auto_register_probe_classes()
