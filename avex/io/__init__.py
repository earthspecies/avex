"""Temporary internal IO module for the public API.

This package provides a small, self-contained IO layer that mirrors the subset of the
``esp_data.io`` interface used by the public API. This allows the package to be
installed and used without requiring ``esp-data`` as a dependency.

Development workflows that rely on ``run_train.py`` or ``run_evaluate.py`` can
continue to use ``esp_data.io`` directly once the optional ``dev`` dependencies are
installed.
"""

from __future__ import annotations

import logging

from .file_utils import exists, rm
from .filesystem import filesystem_from_path
from .paths import AnyPathT, PureCloudPath, PureGSPath, PureR2Path, PureS3Path, anypath

logger = logging.getLogger(__name__)

# Compatibility check: if esp_data.io is available, emit an informational log so that
# environments with the full development stack are aware that the public API uses an
# internal IO shim rather than esp_data.io directly.
try:  # pragma: no cover - purely informational
    import esp_data.io as _esp_io  # type: ignore[import-not-found]

    if hasattr(_esp_io, "anypath"):
        logger.info(
            "esp_data.io is installed; the public API will use "
            "avex.io for IO operations. "
            "Development scripts may still rely on esp_data.io directly.",
        )
except Exception:  # noqa: BLE001 - defensive import guard
    # esp_data is not installed; this is the expected configuration for API-only use.
    pass

__all__ = [
    "AnyPathT",
    "PureCloudPath",
    "PureGSPath",
    "PureR2Path",
    "PureS3Path",
    "anypath",
    "filesystem_from_path",
    "exists",
    "rm",
]
