"""
Checkpoint management for training.

This module handles saving and loading of model checkpoints, including
model state, optimizer state, scheduler state, and training metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.nn as nn
from esp_data.io.paths import GSPath, R2Path, anypath

from representation_learning.configs import RunConfig
from representation_learning.training.distributed import is_main_process
from representation_learning.utils.experiment_logger import (
    get_active_mlflow_run_name,
)

if TYPE_CHECKING:
    from representation_learning.utils.experiment_logger import (
        ExperimentLogger,
    )
from representation_learning.utils.experiment_tracking import (
    save_experiment_metadata,
)

logger = logging.getLogger(__name__)

CloudPathT = GSPath | R2Path


class CheckpointManager:
    """Manages checkpoint saving and loading."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        checkpoint_freq: int = 1,
        experiment_logger: Optional["ExperimentLogger"] = None,
        run_config: Optional[RunConfig] = None,
    ) -> None:
        """Initialize checkpoint manager.

        Parameters
        ----------
        model_dir : Union[str, Path]
            Directory to save checkpoints
        checkpoint_freq : int, optional
            Frequency of checkpointing (in epochs), by default 1
        experiment_logger : Optional[ExperimentLogger], optional
            Experiment logger for saving metadata, by default None
        run_config : Optional[RunConfig], optional
            Run configuration for metadata, by default None
        """
        self.model_dir = anypath(model_dir)
        self.checkpoint_freq = checkpoint_freq
        self.experiment_logger = experiment_logger
        self.run_config = run_config

        # Ensure directory exists
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except AttributeError:
            # Some CloudPath objects don't implement mkdir
            pass

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        best_val_acc: float,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save a checkpoint of the model and training state.

        Parameters
        ----------
        model : nn.Module
            Model to save (should be unwrapped if using DDP)
        optimizer : torch.optim.Optimizer
            Optimizer state to save
        scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
            Scheduler state to save
        scaler : Optional[torch.cuda.amp.GradScaler]
            AMP scaler state to save
        epoch : int
            Current epoch number
        best_val_acc : float
            Best validation accuracy so far
        is_best : bool, optional
            Whether this is the best model, by default False
        is_final : bool, optional
            Whether this is the final checkpoint, by default False
        """
        if not is_main_process():
            return  # Only main process saves checkpoints

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_val_acc": best_val_acc,
        }

        # Determine filename
        if is_final:
            filename = "final_model.pt"
        elif is_best:
            filename = "best_model.pt"
        elif epoch % self.checkpoint_freq == 0:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"
        else:
            return  # Don't save if not periodic, best, or final

        # Determine base directory
        if isinstance(self.model_dir, CloudPathT):
            base_dir = self.model_dir
        elif self.experiment_logger is not None and hasattr(
            self.experiment_logger, "log_dir"
        ):
            base_dir = Path(self.experiment_logger.log_dir)
        else:
            base_dir = Path(self.model_dir)

        # Ensure directory exists
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except AttributeError:
            pass  # Cloud paths may not implement mkdir

        ckpt_path = base_dir / filename

        # Save checkpoint
        if isinstance(ckpt_path, CloudPathT):
            with ckpt_path.open("wb") as f:
                torch.save(checkpoint, f)
        else:
            torch.save(checkpoint, ckpt_path)

        logger.info("Saved checkpoint → %s", ckpt_path)

        # Save metadata
        self._save_metadata(base_dir, filename, is_best, is_final)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint and restore training state.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file
        model : nn.Module
            Model to load state into (should be unwrapped if using DDP)
        optimizer : torch.optim.Optimizer
            Optimizer to load state into
        scheduler : Optional[torch.optim.lr_scheduler._LRScheduler], optional
            Scheduler to load state into, by default None
        scaler : Optional[torch.cuda.amp.GradScaler], optional
            AMP scaler to load state into, by default None

        Returns
        -------
        Dict[str, Any]
            Dictionary containing loaded training state (epoch, best_val_acc, etc.)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(
                f"Checkpoint file not found: {checkpoint_path}. "
                f"Starting training from scratch."
            )
            return {"start_epoch": 1, "best_val_acc": 0.0}

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load model state
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model state from {checkpoint_path}")
        except Exception as e:
            logger.error(
                f"Error loading model state_dict: {e}. "
                f"Model weights might be incompatible."
            )
            return {"start_epoch": 1, "best_val_acc": 0.0}

        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Loaded optimizer state.")
        except Exception as e:
            logger.warning(
                f"Could not load optimizer state: {e}. "
                f"Optimizer will start from scratch."
            )

        # Load scheduler state
        if (
            scheduler
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"]
        ):
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Loaded scheduler state.")
            except Exception as e:
                logger.warning(
                    f"Could not load scheduler state: {e}. "
                    f"Scheduler will start from scratch."
                )

        # Load scaler state
        if (
            scaler
            and "scaler_state_dict" in checkpoint
            and checkpoint["scaler_state_dict"]
        ):
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                logger.info("Loaded AMP scaler state.")
            except Exception as e:
                logger.warning(
                    f"Could not load AMP scaler state: {e}. "
                    f"Scaler will start from scratch."
                )

        # Extract training state
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)

        logger.info(
            f"Resuming training from epoch {start_epoch} with best validation "
            f"accuracy {best_val_acc:.4f}"
        )

        return {
            "start_epoch": start_epoch,
            "best_val_acc": best_val_acc,
        }

    def _save_metadata(
        self,
        base_dir: Path,
        checkpoint_name: str,
        is_best: bool,
        is_final: bool,
    ) -> None:
        """Save experiment metadata alongside checkpoint."""
        if (
            self.experiment_logger
            and self.experiment_logger.backend == "mlflow"
            and self.run_config
            and not self.run_config.run_name
        ):
            self.run_config.run_name = get_active_mlflow_run_name(
                self.experiment_logger
            )

        try:
            save_experiment_metadata(
                output_dir=base_dir,
                config=self.run_config,
                checkpoint_name=checkpoint_name,
                metrics=(
                    self.experiment_logger.last_metrics
                    if (
                        self.experiment_logger
                        and hasattr(self.experiment_logger, "last_metrics")
                    )
                    else {}
                ),
                is_best=is_best,
                is_final=is_final,
            )
            logger.info("Saved metadata for checkpoint %s", checkpoint_name)
        except Exception as e:
            logger.warning(f"Could not save metadata: {e}")
