"""
General utility functions for the representation learning package.

This module contains utility functions that are used across multiple modules.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import torch

from avex.io import AnyPathT, PureCloudPath, anypath, filesystem_from_path

logger = logging.getLogger(__name__)

CACHE_META_SUFFIX = ".avex_cache_meta.json"


def _cache_meta_path(cache_path: Path) -> Path:
    """Return the sidecar metadata path for a cached file.

    Returns
    -------
    Path
        Sidecar metadata path for ``cache_path``.
    """
    return cache_path.with_name(cache_path.name + CACHE_META_SUFFIX)


def _read_cache_meta(cache_path: Path) -> dict[str, Any] | None:  # noqa: ANN401
    """Read cache sidecar metadata if present.

    Returns
    -------
    dict[str, Any] | None
        Parsed metadata if the sidecar file exists and can be read, otherwise
        ``None``.
    """
    meta_path = _cache_meta_path(cache_path)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Failed to read cache meta %s: %s", meta_path, e)
        return None


def _write_cache_meta(cache_path: Path, meta: dict[str, Any]) -> None:  # noqa: ANN401
    """Write cache sidecar metadata."""
    meta_path = _cache_meta_path(cache_path)
    try:
        meta_path.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Failed to write cache meta %s: %s", meta_path, e)


def _remote_version_token(fs: Any, path: AnyPathT) -> str | None:  # noqa: ANN401
    """Best-effort remote version token without downloading.

    Uses fsspec filesystem metadata (`fs.info`) when available. Different backends
    expose different fields; we normalize a token from whichever stable identifiers
    are present (e.g. etag, md5/crc32c/sha256, generation/versionId).

    Returns
    -------
    str | None
        A stable-ish version token derived from remote metadata when available,
        otherwise ``None``.
    """
    try:
        info = fs.info(str(path))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Failed to stat remote path %s: %s", path, e)
        return None

    if not isinstance(info, dict):
        return None

    candidates: list[str] = []
    for key in (
        "etag",
        "ETag",
        "md5",
        "md5Hash",
        "crc32c",
        "sha256",
        "generation",
        "versionId",
        "last_modified",
        "mtime",
        "size",
    ):
        if key in info and info[key] is not None:
            candidates.append(f"{key}={info[key]}")

    if not candidates:
        return None

    return "|".join(candidates)


def _download_atomically(fs: Any, src: AnyPathT, dst: Path) -> None:  # noqa: ANN401
    """Download `src` to `dst` via a temp file + atomic rename.

    This avoids TOCTOU races (check-then-download) and prevents partially written
    cache files from being treated as valid on subsequent runs.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=dst.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        fs.get(str(src), str(tmp_path))
        tmp_path.replace(dst)  # atomic on POSIX (same filesystem)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Failed to clean up temp file %s: %s", tmp_path, e)
        raise


def _safe_cache_path(cache_root: Path, bucket: str, filename: str) -> Path:
    """Build a cache path without trusting user-controlled segments.

    Parameters
    ----------
    cache_root:
        Root directory for cached files.
    bucket:
        Bucket / repo identifier from the URI.
    filename:
        Target cached filename.

    Returns
    -------
    Path
        A cache path guaranteed (after resolve) to live under ``cache_root``.

    Raises
    ------
    ValueError
        If the resolved cache path would fall outside ``cache_root``.
    """
    bucket_digest = hashlib.sha256(bucket.encode("utf-8")).hexdigest()[:16]
    candidate = cache_root / bucket_digest / filename

    # Defense-in-depth: ensure the resolved path stays within cache_root.
    root_resolved = cache_root.resolve()
    candidate_resolved = candidate.resolve()
    if root_resolved not in candidate_resolved.parents and candidate_resolved != root_resolved:
        msg = f"Unsafe cache path resolved outside cache_root: {candidate_resolved}"
        raise ValueError(msg)

    return candidate


def _get_local_path_for_cloud_file(
    path: AnyPathT,
    fs: Any,  # noqa: ANN401
    cache_mode: Literal["none", "use", "force"],
) -> Path | None:
    """Get a local file path for a cloud file, handling caching if needed.

    For cloud paths with caching enabled, downloads to cache directory.
    For cloud paths without caching, returns None (caller should handle directly).
    For local paths, returns the Path object directly.

    Parameters
    ----------
    path : AnyPathT
        Path to the file (local or cloud)
    fs : Any
        Filesystem object for the path
    cache_mode : Literal["none", "use", "force"]
        Cache mode for cloud files

    Returns
    -------
    Path | None
        Local file path if available, None if should read directly from cloud
    """
    is_cloud_path = isinstance(path, PureCloudPath)

    if is_cloud_path:
        if cache_mode in ["use", "force"]:
            cache_root = (
                Path(os.environ["ESP_CACHE_HOME"]) if "ESP_CACHE_HOME" in os.environ else Path.home() / ".cache" / "esp"
            )

            # Avoid collisions across repos / buckets / nested paths by hashing the full URI.
            # This keeps caching predictable even if different repos share the same filename.
            uri = str(path)
            digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
            suffix = Path(path.name).suffix
            cached_name = f"{Path(path.name).stem}-{digest}{suffix}"

            cache_path = _safe_cache_path(cache_root, path.bucket, cached_name)

            if not cache_path.exists() or cache_mode == "force":
                download_msg = (
                    "Force downloading" if cache_mode == "force" else "Cache file does not exist, downloading"
                )
                logger.info(f"{download_msg} to {cache_path}...")
                try:
                    _download_atomically(fs, path, cache_path)
                except (OSError, PermissionError) as e:
                    logger.warning(
                        "Caching is enabled but cache directory is not writable (%s); "
                        "falling back to direct cloud read for %s.",
                        e,
                        path,
                    )
                    return None
                token = _remote_version_token(fs, path)
                if token is not None:
                    _write_cache_meta(
                        cache_path,
                        {"remote_version_token": token, "source_uri": str(path)},
                    )
            else:
                # Cache exists. If we can cheaply validate remote metadata, do so.
                cached_meta = _read_cache_meta(cache_path)
                cached_token = cached_meta.get("remote_version_token") if isinstance(cached_meta, dict) else None
                remote_token = _remote_version_token(fs, path)

                if cached_token is None or remote_token is None:
                    logger.info(
                        "Cannot validate cache for %s (cached_token=%s, remote_token=%s); using local cache.",
                        path,
                        cached_token is not None,
                        remote_token is not None,
                    )
                elif cached_token != remote_token:
                    logger.info(
                        "Remote object changed for %s; re-downloading to refresh cache.",
                        path,
                    )
                    try:
                        _download_atomically(fs, path, cache_path)
                    except (OSError, PermissionError) as e:
                        logger.warning(
                            "Caching is enabled but cache directory is not writable (%s); "
                            "falling back to direct cloud read for %s.",
                            e,
                            path,
                        )
                        return None
                    _write_cache_meta(
                        cache_path,
                        {"remote_version_token": remote_token, "source_uri": str(path)},
                    )
                else:
                    logger.debug(f"Found {cache_path}, using local cache.")
            return cache_path
        else:
            # No caching - return None to indicate direct cloud read
            return None
    else:
        # Local path
        return Path(path)


def _load_torch(
    path: AnyPathT,
    fs: Any,  # noqa: ANN401
    cache_mode: Literal["none", "use", "force"],
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Load a PyTorch checkpoint file using torch.load().

    Handles both local and cloud paths. For cloud paths, can optionally cache
    downloaded files locally or read directly from cloud storage.

    Parameters
    ----------
    path : AnyPathT
        Path to the checkpoint file (local or cloud)
    fs : Any
        Filesystem object for the path
    cache_mode : Literal["none", "use", "force"]
        Cache mode for cloud files
    **kwargs : Any
        Additional keyword arguments passed to torch.load()

    Returns
    -------
    Any
        The object loaded from the file using torch.load()
    """
    local_path = _get_local_path_for_cloud_file(path, fs, cache_mode)

    # Load the file
    if local_path is not None:
        # Local file (cached or original)
        with open(local_path, "rb") as opened_file:
            # Explicitly set weights_only=False for model checkpoints
            # Model checkpoints contain state dicts and other objects, not just weights
            # This suppresses the FutureWarning while maintaining functionality
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return torch.load(opened_file, **kwargs)
    else:
        # Cloud path - read directly
        with fs.open(str(path), "rb") as opened_file:
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return torch.load(io.BytesIO(opened_file.read()), **kwargs)


def _load_safetensor(
    path: AnyPathT,
    fs: Any,  # noqa: ANN401
    cache_mode: Literal["none", "use", "force"],
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:  # noqa: ANN401
    """Load a safetensors file using safetensors.torch.load_file().

    Handles both local and cloud paths. For cloud paths without caching,
    uses a temporary file context manager to ensure proper cleanup.

    Parameters
    ----------
    path : AnyPathT
        Path to the safetensors file (local or cloud)
    fs : Any
        Filesystem object for the path
    cache_mode : Literal["none", "use", "force"]
        Cache mode for cloud files
    **kwargs : Any
        Additional keyword arguments. Supports 'device' and 'map_location'
        for loading tensors on specific device.

    Returns
    -------
    dict[str, Any]
        Dictionary with "model_state_dict" key containing the state dict.
        This format is compatible with checkpoint loading code.
    """
    from safetensors.torch import load_file

    # Extract device/map_location from kwargs if provided (for safetensors)
    # map_location is the standard PyTorch parameter, device is for safetensors
    # Priority: map_location > device > default "cpu"
    map_location = kwargs.pop("map_location", None)
    device = kwargs.pop("device", None)
    if map_location is not None:
        device = map_location
    elif device is None:
        device = "cpu"

    local_path = _get_local_path_for_cloud_file(path, fs, cache_mode)

    if local_path is not None:
        # Local file (cached or original)
        logger.debug(f"Loading safetensors file: {local_path}")
        state_dict = load_file(str(local_path), device=device)

        # Wrap in "model_state_dict" for compatibility with checkpoint loading code
        return {"model_state_dict": state_dict}
    else:
        # No caching - use temp file with context manager for automatic cleanup
        # (safetensors requires a local file, so we must download to temp)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            fs.get(str(path), str(tmp_path))
            logger.debug(f"Loading safetensors file: {tmp_path}")
            state_dict = load_file(str(tmp_path), device=device)

            # Wrap in "model_state_dict" for compatibility with checkpoint loading code
            return {"model_state_dict": state_dict}
        finally:
            # Clean up temp file
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception as e:
                    logger.debug(f"Could not clean up temp file {tmp_path}: {e}")


def universal_torch_load(
    f: str | os.PathLike | AnyPathT,
    *,
    cache_mode: Literal["none", "use", "force"] = "use",
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """
    Wrapper function for torch.load that can handle cloud and local paths.

    This function provides a convenient way to load PyTorch objects from both local and
    cloud storage paths (GCS, R2, etc.). For cloud paths, it can optionally cache the
    downloaded files locally to avoid repeated downloads.

    Supports both PyTorch checkpoints (.pt, .pth) and safetensors files (.safetensors).
    For safetensors files, automatically uses safetensors.torch.load_file() instead of
    torch.load(). Safetensors files are loaded as state dictionaries (wrapped in
    "model_state_dict" key for compatibility with checkpoint loading code).

    The cache location is determined by:
    1. The ESP_CACHE_HOME environment variable if set
    2. Otherwise defaults to ~/.cache/esp/

    Args:
        f: File-like object, string or PathLike object.
           Can be a local path or a cloud path (starting with 'gs://', 'r2://', etc.).
        cache_mode (str, optional): Cache mode for cloud files. Options are:
            "none": No caching (use cloud storage directly)
            "use": Use cache if available, download if not
            "force": Force redownload even if cache exists
            Defaults to "use".
        **kwargs: Additional keyword arguments passed to torch.load() or safetensors.torch.load_file().
                  For safetensors, supports 'device' parameter to load tensors on specific device.

    Returns:
        The object loaded from the file. For safetensors files, returns a dict with
        "model_state_dict" key containing the state dict. For PyTorch checkpoints,
        returns the object as loaded by torch.load().
    """
    path = anypath(f)
    fs = filesystem_from_path(path)

    # Check if this is a safetensors file (by extension)
    path_str = str(path)
    is_safetensors = path_str.endswith(".safetensors")

    if is_safetensors:
        return _load_safetensor(path, fs, cache_mode, **kwargs)
    else:
        return _load_torch(path, fs, cache_mode, **kwargs)


# -------------------------------------------------------------------- #
#  Checkpoint sanitiser helper                                         #
# -------------------------------------------------------------------- #


def _process_state_dict(state_dict: dict, keep_classifier: bool = False, drop_model_prefix: bool = True) -> dict:
    """Process state dict to handle common prefixes and optionally remove
    classifier layers.

    This function handles common checkpoint formats by:
    1. Extracting model state dict if wrapped
    2. Removing common prefixes like 'module.' and 'model.'
    3. Optionally removing classifier layers for backbone loading

    Parameters
    ----------
    state_dict : dict
        The state dictionary to process
    keep_classifier : bool, default False
        If True, keep classifier/head layers in the output.
        If False (default), remove classifier layers for backbone loading.
    drop_model_prefix : bool, default True
        If True, strip leading ``model.`` from parameter names (common in
        DDP/lightning checkpoints). Set to False when the target model's
        parameters already include the ``model.`` prefix to avoid mismatches.

    Returns
    -------
    dict
        Processed state dictionary with prefixes removed and optionally
        classifier layers dropped.
    """
    # Extract model state dict if wrapped
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    # Conditionally drop common classifier parameter names (different wrappers)
    # These are specific known problematic keys that should be removed when
    # keep_classifier=False
    if not keep_classifier:
        state_dict.pop("classifier.weight", None)
        state_dict.pop("classifier.bias", None)
        state_dict.pop("model.classifier.1.weight", None)
        state_dict.pop("model.classifier.1.bias", None)

    # Create a new state dict with processed keys
    processed_dict = {}

    for key, value in state_dict.items():
        # Remove common prefixes
        processed_key = key
        if processed_key.startswith("module."):
            processed_key = processed_key[7:]  # Remove "module."
        elif drop_model_prefix and processed_key.startswith("model."):
            processed_key = processed_key[6:]  # Remove "model."

        # Conditionally skip classifier layers based on keep_classifier parameter
        if not keep_classifier and any(
            term in processed_key.lower() for term in ["classifier", "head", "classification", "classification_head"]
        ):
            continue

        processed_dict[processed_key] = value

    return processed_dict
