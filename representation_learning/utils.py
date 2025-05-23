"""
exp_logger.py
~~~~~~~~~~~~~
Backend‑agnostic experiment logger supporting **MLflow** and **Weights & Biases**.

Usage
-----
from representation_learning.utils.exp_logger import get_logger

log = get_logger(cfg)           # picks mlflow / wandb / no‑op
log.log_params({...})           # hyper‑parameters
log.log_metrics({...}, step=1)  # metrics per epoch
log.finalize()                  # flush / close
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import Any, Dict, Optional, Union, Literal
import os
from pathlib import Path

from functools import lru_cache
import cloudpathlib
from google.cloud.storage.client import Client
import torch


from representation_learning.configs import RunConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Core class
# --------------------------------------------------------------------------- #
class ExperimentLogger:
    """
    Wraps a single backend instance (MLflow or W&B) or acts as a no‑op.

    Methods
    -------
    log_params(dict)
    log_metrics(dict, step, split="train")
    finalize()
    """

    # ------------------------ construction helpers ------------------------ #
    @classmethod
    def from_config(cls, cfg: RunConfig) -> "ExperimentLogger":
        backend = str(getattr(cfg, "logging", "none")).lower()
        run_name = getattr(cfg, "run_name", None)

        if backend == "mlflow":
            return cls._build_mlflow(run_name)

        if backend in {"wandb", "wb"}:
            project = getattr(cfg, "wandb_project", "audio‑experiments")
            return cls._build_wandb(project, run_name)

        # anything else → no‑op
        return cls._build_none()

    @classmethod
    def _build_mlflow(cls, run_name: Optional[str]) -> "ExperimentLogger":
        try:
            mlflow = importlib.import_module("mlflow")
        except ModuleNotFoundError:
            logger.warning("mlflow not installed – logging disabled.")
            return cls(backend="none")

        mlflow.set_tracking_uri(uri="http://100.89.114.62:8080")
        mlflow.start_run(run_name=run_name)
        logger.info("MLflow run started (%s).", run_name)
        return cls(backend="mlflow", handle=mlflow)

    @classmethod
    def _build_wandb(cls, project: str, run_name: Optional[str]) -> "ExperimentLogger":
        try:
            wandb = importlib.import_module("wandb")
        except ModuleNotFoundError:
            logger.warning("wandb not installed – logging disabled.")
            return cls(backend="none")

        handle = wandb.init(project=project, name=run_name, config={})
        logger.info("Weights & Biases run initialised (%s).", run_name)
        return cls(backend="wandb", handle=handle)

    @classmethod
    def _build_none(cls) -> "ExperimentLogger":
        logger.info("Experiment logging disabled (backend=none).")
        return cls(backend="none")

    # ------------------------------ dunder ------------------------------- #
    def __init__(self, *, backend: str, handle: Union[ModuleType, None] = None) -> None:
        """Initialize the experiment logger.

        Parameters
        ----------
        backend : str
            The logging backend to use ("mlflow", "wandb", or "none")
        handle : Union[ModuleType, None]
            The backend-specific handle (mlflow module, wandb run, or None)
        """
        self.backend = backend
        self.handle = handle  # mlflow module OR wandb run OR None

    # ------------------------------ API ---------------------------------- #
    def log_params(self, params: Dict[str, Any]) -> None:
        if self.backend == "mlflow":
            self.handle.log_params(params)  # type: ignore[attr-defined]
        elif self.backend == "wandb":
            self.handle.config.update(params, allow_val_change=True)  # type: ignore[attr-defined]

    def log_metrics(
        self,
        metrics: Dict[str, float],
        *,
        step: int,
        split: str = "train",
    ) -> None:
        if self.backend == "mlflow":
            for k, v in metrics.items():
                self.handle.log_metric(f"{split}_{k}", v, step=step)  # type: ignore[attr-defined]

        elif self.backend == "wandb":
            self.handle.log({f"{split}/{k}": v for k, v in metrics.items()}, step=step)  # type: ignore[attr-defined]

    def finalize(self) -> None:
        if self.backend == "mlflow":
            self.handle.end_run()  # type: ignore[attr-defined]
        elif self.backend == "wandb":
            self.handle.finish()  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
#  Convenience getter (singleton per process)
# --------------------------------------------------------------------------- #
_instance: ExperimentLogger | None = None


def get_logger(cfg: RunConfig) -> ExperimentLogger:
    """Create (or return existing) ExperimentLogger for this process.

    Parameters
    ----------
    cfg : RunConfig
        Configuration object containing logging settings

    Returns
    -------
    ExperimentLogger
        Logger instance for the current process
    """
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        _instance = ExperimentLogger.from_config(cfg)
    return _instance

@lru_cache(maxsize=1)
def _get_client():
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

    def __init__(self, client_path, client=_get_client()):
        super().__init__(client_path, client=client)


def is_gcs_path(path: Union[str, os.PathLike]) -> bool:
    return str(path).startswith("gs://")


def universal_torch_load(
    f: str | os.PathLike | GSPath,
    *,
    cache_mode: Literal["none", "use", "force"] = "none",
    **kwargs,
) -> Any:
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
                logger.info(
                    f"{'Force downloading' if cache_mode == 'force' else 'Cache file does not exist, downloading'} to {cache_path}..."
                )
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
