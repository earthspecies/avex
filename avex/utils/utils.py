"""
General utility functions for the representation learning package.

This module contains utility functions that are used across multiple modules.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import torch

from avex.io import AnyPathT, PureCloudPath, anypath, filesystem_from_path

logger = logging.getLogger(__name__)


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
            # Use cache
            if "ESP_CACHE_HOME" in os.environ:
                cache_path = Path(os.environ["ESP_CACHE_HOME"]) / path.name
            else:
                cache_path = Path.home() / ".cache" / "esp" / path.name

            if not cache_path.exists() or cache_mode == "force":
                download_msg = (
                    "Force downloading" if cache_mode == "force" else "Cache file does not exist, downloading"
                )
                logger.info(f"{download_msg} to {cache_path}...")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                fs.get(str(path), str(cache_path))
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
    cache_mode: Literal["none", "use", "force"] = "none",
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
            Defaults to "none".
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
