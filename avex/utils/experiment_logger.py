"""
exp_logger.py
~~~~~~~~~~~~~
Backend‑agnostic experiment logger supporting **MLflow** and **Weights & Biases**.

Usage
-----
from avex.utils.exp_logger import get_logger

log = get_logger(cfg)           # picks mlflow / wandb / no‑op
log.log_params({...})           # hyper‑parameters
log.log_metrics({...}, step=1)  # metrics per epoch
log.finalize()                  # flush / close
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Union

from avex.configs import RunConfig

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
        logging_uri = getattr(cfg, "logging_uri", None)
        run_name = getattr(cfg, "run_name", None)

        if backend == "mlflow":
            return cls._build_mlflow(run_name, logging_uri=logging_uri)

        if backend in {"wandb", "wb"}:
            project = getattr(cfg, "wandb_project", "audio‑experiments")
            return cls._build_wandb(project, run_name)

        # anything else → no‑op
        return cls._build_none()

    @classmethod
    def _build_mlflow(cls, run_name: Optional[str], logging_uri: Optional[str]) -> "ExperimentLogger":
        try:
            mlflow = importlib.import_module("mlflow")
        except ModuleNotFoundError:
            logger.warning("mlflow not installed – logging disabled.")
            return cls(backend="none")

        mlflow.set_tracking_uri(uri=logging_uri or "http://localhost:5000")
        mlflow.start_run(run_name=run_name)
        logger.info("MLflow run started (%s).", run_name)
        return cls(backend="mlflow", handle=mlflow)

    @classmethod
    def _handle_wandb_auth(cls, wandb_module: ModuleType) -> None:
        """Handle W&B authentication using environment variables or secrets file.

        Parameters
        ----------
        wandb_module : ModuleType
            The imported wandb module
        """
        # First try environment variable
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            logger.info("Using WANDB_API_KEY from environment variable")
            wandb_module.login(key=api_key)
            return

        # Try secrets file
        secrets_file = Path(".secrets.yaml")
        if secrets_file.exists():
            try:
                import yaml

                with secrets_file.open("r") as f:
                    secrets = yaml.safe_load(f)

                api_key = secrets.get("wandb_api_key")
                if api_key:
                    logger.info("Using WANDB_API_KEY from .secrets.yaml")
                    wandb_module.login(key=api_key)
                    return
            except Exception as e:
                logger.warning(f"Failed to read secrets file: {e}")

        # Let wandb handle authentication naturally (might already be logged in)
        try:
            # Check if already logged in
            if wandb_module.api.api_key:
                logger.info("Using existing wandb authentication")
                return
        except Exception:
            pass

        # Try to login without key (will use wandb's default behavior)
        try:
            wandb_module.login()
            logger.info("Successfully authenticated with wandb")
        except Exception as e:
            logger.warning(
                f"wandb authentication failed: {e}. "
                "Consider setting WANDB_API_KEY environment variable or "
                "creating a .secrets.yaml file with 'wandb_api_key: your_key_here'"
            )

    @classmethod
    def _build_wandb(cls, project: str, run_name: Optional[str]) -> "ExperimentLogger":
        try:
            wandb = importlib.import_module("wandb")
        except ModuleNotFoundError:
            logger.warning("wandb not installed – logging disabled.")
            return cls(backend="none")

        # Handle authentication before initializing run
        cls._handle_wandb_auth(wandb)

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


def get_active_mlflow_run_name(logger: ExperimentLogger) -> str:
    """Get active run's name from mflow backend.

    Returns
    -------
    run_name: str
        The active run's name
    """
    run = logger.handle.active_run()
    return run.info.run_name


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
