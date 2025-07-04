"""
General utility functions for the representation learning package.

This module contains utility functions that are used across multiple modules,
including GCS path handling and universal file loading functionality.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Union

import cloudpathlib
import torch
from google.cloud.storage.client import Client

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> cloudpathlib.GSClient:
    """Get a cached Google Cloud Storage client.

    Returns
    -------
    cloudpathlib.GSClient
        A Google Cloud Storage client instance.
    """
    return cloudpathlib.GSClient(storage_client=Client())


class GSPath(cloudpathlib.GSPath):
    """
    A wrapper for the GSPath class that provides a default client to the constructor.
    This is necessary due to a bug in cloudpathlib (v0.20.0) which assumes that the
    GOOGLE_APPLICATION_CREDENTIALS environment variable always points to a service
    account. This assumption is incorrect when using Workload Identity Federation, which
    we in our Github Action. Here, we fallback to the actual Google library for a
    default client that handles this correctly.

    For more details, see: https://github.com/drivendataorg/cloudpathlib/issues/390
    """

    def __init__(
        self,
        client_path: str | os.PathLike,
        client: cloudpathlib.GSClient | None = None,
    ) -> None:
        if client is None:
            client = _get_client()
        super().__init__(client_path, client=client)


def is_gcs_path(path: Union[str, os.PathLike]) -> bool:
    """Check if a path is a Google Cloud Storage path.

    Parameters
    ----------
    path : Union[str, os.PathLike]
        The path to check.

    Returns
    -------
    bool
        True if the path is a GCS path (starts with 'gs://'), False otherwise.
    """
    return str(path).startswith("gs://")


def universal_torch_load(
    f: str | os.PathLike | GSPath,
    *,
    cache_mode: Literal["none", "use", "force"] = "none",
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """
    Wrapper function for torch.load that can handle GCS paths.

    This function provides a convenient way to load PyTorch objects from both local and
    Google Cloud Storage (GCS) paths. For GCS paths, it can optionally caches the
    downloaded files locally to avoid repeated downloads.

    The cache location is determined by:
    1. The ESP_CACHE_HOME environment variable if set
    2. Otherwise defaults to ~/.cache/esp/

    Args:
        f: File-like object, string or PathLike object.
           Can be a local path or a GCS path (starting with 'gs://').
        cache_mode (str, optional): Cache mode for GCS files. Options are:
            "none": No caching (use bucket directly)
            "use": Use cache if available, download if not
            "force": Force redownload even if cache exists
            Defaults to "none".
        **kwargs: Additional keyword arguments passed to torch.load().

    Returns:
        The object loaded from the file using torch.load.

    Raises:
        IsADirectoryError: If the GCS path points to a directory instead of a file.
        FileNotFoundError: If the local file does not exist.
    """
    if is_gcs_path(f):
        gs_path = GSPath(str(f))
        if gs_path.is_dir():
            raise IsADirectoryError(f"Cannot load a directory: {f}")

        if cache_mode in ["use", "force"]:
            if "ESP_CACHE_HOME" in os.environ:
                cache_path = Path(os.environ["ESP_CACHE_HOME"]) / gs_path.name
            else:
                cache_path = Path.home() / ".cache" / "esp" / gs_path.name

            if not cache_path.exists() or cache_mode == "force":
                download_msg = (
                    "Force downloading"
                    if cache_mode == "force"
                    else "Cache file does not exist, downloading"
                )
                logger.info(f"{download_msg} to {cache_path}...")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                gs_path.download_to(cache_path)
            else:
                logger.debug(f"Found {cache_path}, using local cache.")
            f = cache_path
        else:
            f = gs_path
    else:
        f = Path(f)
        if not f.exists():
            raise FileNotFoundError(f"File does not exist: {f}")

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
