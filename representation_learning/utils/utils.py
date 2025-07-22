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

    if path.is_cloud:
        if path.is_dir():
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
                path.download_to(cache_path)
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


# -------------------------------------------------------------------- #
#  EfficientNet checkpoint helper                                     #
# -------------------------------------------------------------------- #


def sanitize_efficientnet_state_dict(state_dict: dict) -> dict:
    """Prepare an EfficientNet checkpoint for loading into audio encoder.

    This helper removes common wrapping layers (``model.``, ``module.``),
    strips the classifier head to avoid shape mismatches and returns a clean
    mapping that can be passed directly to ``load_state_dict``.

    The function is intentionally *loss-less* for backbone parameters – only
    obvious classifier keys are dropped.  Prefix removal is performed in a
    loop so nested wrappers (e.g. ``module.model.features…``) are handled.

    Parameters
    ----------
    state_dict : dict
        Original state-dict (possibly wrapped).

    Returns
    -------
    dict
        Sanitised state-dict ready for ``load_state_dict``.
    """

    # Unwrap common containers first
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    clean_sd: dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        # Remove DistributedDataParallel prefix
        if k.startswith("module."):
            k = k[len("module.") :]

        # Remove training wrapper prefix (our scripts save under "model.")
        if k.startswith("model."):
            k = k[len("model.") :]

        # Skip classifier layers (they differ in shape across tasks)
        if k.startswith("classifier"):
            continue

        clean_sd[k] = v

    return clean_sd
