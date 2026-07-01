"""Data module for representation learning.

For development and training workflows, this package integrates with
``alp_data`` to register data transforms. For API-only usage (where
``alp_data`` is not installed), the transforms module is skipped so that
importing :mod:`avex` does not require ``alp_data``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Import transforms to ensure they are registered when alp_data is available.
# For API-only installs (no alp_data), we skip this import to avoid hard
# dependency on the optional alp-data package.
try:
    import alp_data.transforms as _alp_transforms  # type: ignore[import-not-found]  # noqa: F401
except ImportError:
    logger.info(
        "alp_data is not installed; skipping registration of alp_data-based "
        "transforms in avex.data. This is expected for "
        "API-only installations."
    )
else:
    from . import transforms  # noqa: F401
