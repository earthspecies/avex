# New file: avex/data/cloudpathlib_retry_patch.py
"""Robustify *cloudpathlib* downloads during dataset initialisation.

esp-data (and therefore our loaders) rely on ``anypath().read_text`` which, for
GCS/Cloud URLs, maps to ``cloudpathlib.CloudPath.read_text``.  Under heavy
multi-process load this occasionally fails inside Google-cloud-storage’s
``Blob.download_to_filename`` with a *FileNotFoundError* on the final
``os.utime`` call – most likely a race where another worker moves/deletes the
partial file.

This patch wraps *read_text* with a small retry loop that:
1. Catches ``FileNotFoundError``.
2. Deletes the half-baked local cache file if it exists.
3. Waits a short, exponentially-increasing delay.
4. Retries (up to three times by default).

If all retries fail the original exception is re-raised so that upstream code
still aborts rather than silently yielding corrupted data.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable

logger = logging.getLogger(__name__)


def _apply_read_text_retry_patch() -> bool:  # noqa: D401 – simple status helper
    """Monkey-patch :pymeth:`cloudpathlib.cloudpath.CloudPath.read_text`.

    Returns:
        bool: True if patch was applied successfully, False otherwise.
    """
    try:
        from cloudpathlib.cloudpath import CloudPath  # type: ignore
    except Exception as exc:  # pragma: no cover – cloudpathlib missing
        logger.error("cloudpathlib not importable – skipping retry patch: %s", exc)
        return False

    # Idempotency – don’t wrap multiple times
    if getattr(CloudPath.read_text, "_is_retry_wrapped", False):  # type: ignore[attr-defined]
        return True

    original_read_text: Callable[..., str] = CloudPath.read_text  # type: ignore[assignment]

    def _patched_read_text(self: "CloudPath", *args, **kwargs) -> str:  # type: ignore[name-defined]  # noqa: ANN002,ANN003
        max_retries = int(os.environ.get("CLOUDPATHLIB_READ_RETRIES", "3"))
        delay_base = float(os.environ.get("CLOUDPATHLIB_READ_RETRY_DELAY", "1.0"))

        for attempt in range(max_retries):
            try:
                return original_read_text(self, *args, **kwargs)
            except FileNotFoundError as exc:
                if attempt >= max_retries - 1:
                    logger.error(
                        "CloudPath.read_text ultimately failed for %s after %d attempts: %s",
                        self,
                        max_retries,
                        exc,
                    )
                    raise

                logger.warning(
                    "CloudPath.read_text FileNotFound for %s (attempt %d/%d). Retrying…",
                    self,
                    attempt + 1,
                    max_retries,
                )

                # Remove any partially-downloaded file to force a fresh download
                try:
                    local_path = getattr(self, "_local", None)
                    if local_path is not None and local_path.exists():  # type: ignore[attr-defined]
                        local_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception as cleanup_exc:  # pragma: no cover – best-effort
                    logger.debug(
                        "Failed to clean up local cache file %s: %s",
                        local_path,
                        cleanup_exc,
                    )

                time.sleep(delay_base * (attempt + 1))

    # Mark the wrapper so we don’t double-patch
    _patched_read_text._is_retry_wrapped = True  # type: ignore[attr-defined]
    CloudPath.read_text = _patched_read_text  # type: ignore[assignment]

    logger.info(
        "cloudpathlib read_text retry-patch installed (max_retries=%s)",
        os.environ.get("CLOUDPATHLIB_READ_RETRIES", "3"),
    )
    return True


def apply_cloudpathlib_patch() -> bool:  # noqa: D401 – public helper
    """Public entry-point – safe to call multiple times.

    Returns:
        bool: True if patch was applied successfully, False otherwise.
    """
    return _apply_read_text_retry_patch()
