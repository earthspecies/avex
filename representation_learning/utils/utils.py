"""
General utility functions for the representation learning package.

This module contains utility functions that are used across multiple modules.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import torch
from esp_data.io import AnyPathT, anypath
from esp_data.io.paths import PureGSPath, PureR2Path
from esp_data.io.filesystem import filesystem_from_path

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
        The object loaded from the file using torch.load.

    Raises:
        IsADirectoryError: If the cloud path points to a directory instead of a file.
        FileNotFoundError: If the local file does not exist.
    """
    path = anypath(f)

    if isinstance(path, (PureGSPath, PureR2Path)):
        # For cloud paths, use filesystem API for I/O operations
        fs = filesystem_from_path(str(path))
        path_str = str(path)
        
        # Check if it's a directory (cloud paths that end with / are directories)
        if path_str.endswith('/'):
            raise IsADirectoryError(f"Cannot load a directory: {f}")

        if cache_mode in ["use", "force"]:
            if "ESP_CACHE_HOME" in os.environ:
                cache_path = Path(os.environ["ESP_CACHE_HOME"]) / path.name
            else:
                cache_path = Path.home() / ".cache" / "esp" / path.name

            if not cache_path.exists() or cache_mode == "force":
                download_msg = (
                    "Force downloading"
                    if cache_mode == "force"
                    else "Cache file does not exist, downloading"
                )
                logger.info(f"{download_msg} to {cache_path}...")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Use filesystem API to download
                fs.get(path_str, str(cache_path))
            else:
                logger.debug(f"Found {cache_path}, using local cache.")
            f = cache_path
        else:
            f = path
    else:
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {f}")
        f = path

    with open(f, "rb") as opened_file:
        return torch.load(opened_file, **kwargs)


# -------------------------------------------------------------------- #
#  Checkpoint sanitiser helper                                         #
# -------------------------------------------------------------------- #


def _process_state_dict(state_dict: dict) -> dict:
    """Remove classifier layers when loading backbone checkpoints.

    Returns
    -------
    dict
        Processed state dictionary with classifier layers removed.
    """
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Safely drop common classifier parameter names (different wrappers)
    state_dict.pop("classifier.weight", None)
    state_dict.pop("classifier.bias", None)
    state_dict.pop("model.classifier.1.weight", None)
    state_dict.pop("model.classifier.1.bias", None)

    return state_dict
