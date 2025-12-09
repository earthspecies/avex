"""
General utility functions for the representation learning package.

This module contains utility functions that are used across multiple modules.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Literal

import torch

from representation_learning.io import AnyPathT, PureCloudPath, anypath, filesystem_from_path

logger = logging.getLogger(__name__)


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
        **kwargs: Additional keyword arguments passed to torch.load().

    Returns:
        The object loaded from the file using torch.load.g
    """
    path = anypath(f)
    fs = filesystem_from_path(path)

    if isinstance(path, PureCloudPath):
        if cache_mode in ["use", "force"]:
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
            f = cache_path
            fs = filesystem_from_path(f)  # local filesystem
        else:
            f = path
    else:
        f = path

    with fs.open(str(f), "rb") as opened_file:
        # Explicitly set weights_only=False for model checkpoints
        # Model checkpoints contain state dicts and other objects, not just weights
        # This suppresses the FutureWarning while maintaining functionality
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return torch.load(io.BytesIO(opened_file.read()), **kwargs)


# -------------------------------------------------------------------- #
#  Checkpoint sanitiser helper                                         #
# -------------------------------------------------------------------- #


def _process_state_dict(state_dict: dict, keep_classifier: bool = False) -> dict:
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
        elif processed_key.startswith("model."):
            processed_key = processed_key[6:]  # Remove "model."

        # Conditionally skip classifier layers based on keep_classifier parameter
        if not keep_classifier and any(
            term in processed_key.lower() for term in ["classifier", "head", "classification", "classification_head"]
        ):
            continue

        processed_dict[processed_key] = value

    return processed_dict
