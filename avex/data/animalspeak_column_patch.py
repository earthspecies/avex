"""
Patch that trims the AnimalSpeak dataframe to a minimal set of relevant columns.

The patch works exactly like the other dataset patches in
``avex.data``:

1. We monkey-patch
   :pyfunc:`esp_data.datasets.animalspeak.AnimalSpeak._load`.
2. After the original ``_load`` finishes, we drop all columns except a
   hard-coded keep-list.
3. The keep-list is applied defensively – if a listed column is absent we
   skip it — so the patch keeps working when the upstream schema changes
   slightly.

Call :pyfunc:`apply_animalspeak_column_patch` once at import time to activate
the change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from esp_data.datasets.animalspeak import AnimalSpeak

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Hard-coded schema                                                           #
# --------------------------------------------------------------------------- #
# The columns below are needed by the training pipeline (caption generation,
# label transforms, path resolution, etc.).  **Everything else will be dropped**
# as soon as the CSV is loaded.

_KEEP_COLUMNS: List[str] = [
    "local_path",  # relative audio filepath – used by AnimalSpeak._process()
    "caption",  # primary caption (ASR annotated)
    "caption2",  # secondary caption / free-text
    "species_common",  # e.g. "Black-capped Chickadee"
    "canonical_name",  # canonical species name used for labels
    "taxonomic_name",  # scientific name (Genus species)
    "source",  # source of the audio
]

# --------------------------------------------------------------------------- #
#  Patching helpers                                                            #
# --------------------------------------------------------------------------- #

_original_load = None  # will hold reference to the unpatched method


def _patched_load(self: "AnimalSpeak") -> None:  # type: ignore[override] – monkey-patching
    """Replacement for :pymeth:`AnimalSpeak._load`.

    It calls the original ``_load`` implementation, then prunes ``self._data``
    to **only** the columns listed in ``_KEEP_COLUMNS`` (if they exist).
    TODO: remember -- dropping rows with missing canonical_name
    """
    # Call the genuine loader first (populates self._data)
    assert _original_load is not None, "Original _load not stored – patch order bug"
    _original_load(self)

    # Defensive: keep intersection only
    available_cols = [c for c in _KEEP_COLUMNS if c in self._data.columns]
    if not available_cols:
        logger.warning("AnimalSpeak column-patch: none of the expected columns present! Skipping pruning.")
        return

    missing_cols = set(_KEEP_COLUMNS) - set(available_cols)
    if missing_cols:
        logger.info(
            "AnimalSpeak column-patch: missing expected columns %s – continuing with %s",
            sorted(missing_cols),
            available_cols,
        )

    # Use backend-agnostic methods (works with both Polars and Pandas backends)
    self._data = self._data.select_columns(available_cols).dropna(subset=["canonical_name"])

    logger.info(
        "AnimalSpeak column-patch applied => kept %d columns: %s",
        len(available_cols),
        available_cols,
    )


def apply_animalspeak_column_patch() -> bool:  # noqa: D401 – simple status helper
    """Apply the pruning patch once.

    Returns
    -------
    bool
        True on success, False if patch was already applied.
    """
    global _original_load
    try:
        from esp_data.datasets.animalspeak import AnimalSpeak  # type: ignore

        if getattr(AnimalSpeak._load, "__name__", "") == _patched_load.__name__:
            # Already patched
            return True

        _original_load = AnimalSpeak._load  # type: ignore[assignment]
        AnimalSpeak._load = _patched_load  # type: ignore[assignment]
        logger.info("AnimalSpeak column-patch installed successfully")
        return True
    except Exception as exc:  # pragma: no cover – best-effort patch
        logger.error("Failed to apply AnimalSpeak column-patch: %s", exc)
        return False
