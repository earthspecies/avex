"""Data module for representation learning.

For development and training workflows, this package integrates with
``esp_data`` to register data transforms. For API-only usage (where
``esp_data`` is not installed), the transforms module is skipped so that
importing :mod:`representation_learning` does not require ``esp_data``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Import transforms to ensure they are registered when esp_data is available.
# For API-only installs (no esp_data), we skip this import to avoid hard
# dependency on the private esp-data package.
try:
    import esp_data.transforms as _esp_transforms  # type: ignore[import-not-found]  # noqa: F401
except ImportError:
    logger.info(
        "esp_data is not installed; skipping registration of esp_data-based "
        "transforms in representation_learning.data. This is expected for "
        "API-only installations."
    )
else:
    from . import transforms  # noqa: F401
