"""
representation_learning.training.train
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A training loop that supports:

* Multiple training strategies (supervised, CLIP, SSL)
* AMP (fp16 / bf16)
* Gradient‑scaling when using fp16 AMP
* Grad checkpointing for supported models
* Proper no‑grad evaluation
* Checkpointing
* Parameter & metric logging via `ExperimentLogger`
* Distributed training with proper synchronization
* Learning rate scheduling with warmup
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.parallel as parallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from representation_learning.configs import RunConfig
from representation_learning.training.training_strategies import TrainingStrategy, EATSSLStrategy
from representation_learning.training.checkpoint_manager import CheckpointManager
from representation_learning.training.metrics_tracker import MetricsTracker
from representation_learning.training.distributed import (
    is_main_process,
    get_local_device_index,
    cleanup_distributed,
)
from representation_learning.utils import ExperimentLogger

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """
    Modular trainer using strategy pattern for different training modes.
    
    This trainer delegates training logic to strategies and uses separate
    components for checkpointing and metrics tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[GradScaler],
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        strategy: TrainingStrategy,
        checkpoint_manager: CheckpointManager,
        metrics_tracker: MetricsTracker,
        config: RunConfig,
        device: torch.device,
        local_rank: int,
        world_size: int,
        is_distributed: bool,
        exp_logger: Optional[ExperimentLogger] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """Initialize the trainer with all components."""
        # Core components
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.train_dataloader = train_dataloader
        self.eval_dataloader =  eval_dataloader
        self.device = device
        self.config = config
        self.exp_logger = exp_logger
        
        # Distributed training info
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = is_distributed
        
        # Modular components
        self.strategy = strategy
        self.checkpoint_manager = checkpoint_manager
        self.metrics_tracker = metrics_tracker
        
        # Setup model for training
        self.model.to(self.device)
        
        # Enable gradient checkpointing if requested
        if config.training_params.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for memory optimization")
            try:
                self.model.enable_gradient_checkpointing()
            except NotImplementedError as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")
            logger.info("gradient checkpointing enabled successfully")
        
        # Wrap model for distributed training
        self.model = self._wrap_model_for_distributed(self.model)
        
        # AMP setup
        self.amp_enabled = config.training_params.amp
        self.amp_dtype = {
            "fp16": torch.float16, 
            "bf16": torch.bfloat16
        }[config.training_params.amp_dtype]
        
        # Training state
        self.best_val_acc = 0.0
        self.start_epoch = 1
        self.global_updates = 0
        
        # Load checkpoint if specified
        if is_main_process() and resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        # Broadcast training state in distributed setting
        self._broadcast_training_state()
        
        # Log static hyperparameters
        self._log_hyperparameters()

    # ----------------------------- public API ------------------------------ #
    def train(self) -> None:
        """Run the full training loop for the configured number of epochs."""
        try:
            for epoch in range(self.start_epoch, self.config.training_params.train_epochs + 1):
                # Set epoch for distributed samplers
                self._set_epoch_for_samplers(epoch)

                # Run training and validation epochs
                train_loss, train_metrics = self._run_epoch(train=True, epoch=epoch)
                val_loss, val_metrics = self._run_epoch(train=False, epoch=epoch)

                # Extract primary metric for comparison
                primary_metric_name = self.metrics_tracker.primary_metric_name
                train_acc = train_metrics.get(primary_metric_name, 0.0)
                val_acc = val_metrics.get(primary_metric_name, 0.0)

                # Log and save checkpoints on main process only
                if is_main_process():
                    self._log_epoch_results(
                        epoch, train_loss, train_metrics, val_loss, val_metrics
                    )
                    self._handle_checkpointing(epoch, val_acc)

            # Save final checkpoint
            if is_main_process():
                self._save_final_checkpoint()
        finally:
            # Cleanup
            if self.is_distributed:
                cleanup_distributed()
            if is_main_process() and self.exp_logger:
                self.exp_logger.finalize()

    # --------------------------- internal helpers -------------------------- #
    def _run_epoch(self, *, train: bool, epoch: int) -> tuple[float, Dict[str, float]]:
        """
        Iterate once over the loader using the strategy pattern.

        Returns
        -------
        tuple[float, Dict[str, float]] : mean loss, dict of metric values
        """
        loader = self.train_dataloader if train else self.eval_dataloader

        if train:
            self.model.train()
        else:
            self.model.eval()

        # Reset metrics tracker for new epoch
        self.metrics_tracker.reset_epoch_state()



        context_mgr = torch.enable_grad() if train else torch.no_grad()
        is_main = is_main_process()

        with context_mgr:
            iterator = enumerate(loader)
            if is_main:
                iterator = enumerate(
                    tqdm(
                        loader,
                        desc=f"{'Train' if train else 'Eval '} Epoch {epoch}",
                        leave=False,
                    )
                )

            for i, batch in iterator:
                if train:
                    self.global_updates += 1

                # Forward pass using strategy
                result = self._forward_with_strategy(batch)
                
                # Update metrics tracker
                self.metrics_tracker.update_batch_metrics(
                    result.loss, result.metrics_data, result.batch_size, 
                    result.additional_metrics
                )

                if train:
                    # Backward pass and optimization
                    self._backward_step(result.loss)
                    
                    # Update EAT SSL teacher if needed
                    if isinstance(self.strategy, EATSSLStrategy):
                        self.strategy.update_teacher(self.model, self.global_updates)

                # Periodic logging
                if (i + 1) % self.config.training_params.log_steps == 0 or (i + 1) == len(loader):
                    self._log_batch_progress(i + 1, len(loader))

        # Get final epoch metrics
        return self.metrics_tracker.get_epoch_metrics()

    def _forward_with_strategy(self, batch: Dict[str, Any]):
        """Forward pass using the current strategy."""
        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }



        # Forward pass with AMP
        with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            result = self.strategy.forward(self.model, batch)



        return result

    def _backward_step(self, loss: torch.Tensor) -> None:
        """Perform backward pass and optimization step."""
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    def _log_batch_progress(self, step: int, total_steps: int) -> None:
        """Log batch-level progress."""
        if not is_main_process():
            return

        avg_loss, avg_acc = self.metrics_tracker.get_batch_metrics()
        component_metrics = self.metrics_tracker.get_component_metrics()

        # Build log message
        comp_str = ""
        if component_metrics:
            comp_str = "  " + "  ".join(f"{k}={v:.2f}" for k, v in component_metrics.items())

        logger.info(
            f"[LOG] Step {step}/{total_steps}: "
            f"avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.4f}{comp_str}"
        )

        # Log to experiment tracker
        if self.exp_logger:
            metrics_to_log = {
                "loss": avg_loss,
                "acc": avg_acc,
                **component_metrics,
            }
            self.exp_logger.log_metrics(metrics_to_log, step=self.global_updates)

    def _log_epoch_results(
        self, 
        epoch: int, 
        train_loss: float, 
        train_metrics: Dict[str, float], 
        val_loss: float, 
        val_metrics: Dict[str, float]
    ) -> None:
        """Log epoch-level results."""
        current_lr = self.optimizer.param_groups[0]["lr"]
        primary_metric = self.metrics_tracker.primary_metric_name
        
        train_acc = train_metrics.get(primary_metric, 0.0)
        val_acc = val_metrics.get(primary_metric, 0.0)

        # Basic log message
        log_msg = (
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}  "
            f"train_{primary_metric}={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  "
            f"val_{primary_metric}={val_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # Add CLIP-specific metrics if available
        if self.strategy.get_expected_metrics_format() == "clip_accuracy":
            clip_metrics = self.metrics_tracker.get_clip_additional_metrics(self.model)
            if "acc_a2t" in train_metrics and "acc_t2a" in train_metrics:
                log_msg += (
                    f" | train_a2t={train_metrics['acc_a2t']:.4f} "
                    f"train_t2a={train_metrics['acc_t2a']:.4f}"
                )
            if "acc_a2t" in val_metrics and "acc_t2a" in val_metrics:
                log_msg += (
                    f" val_a2t={val_metrics['acc_a2t']:.4f} "
                    f"val_t2a={val_metrics['acc_t2a']:.4f}"
                )

        logger.info(log_msg)

        # Log to experiment tracker
        if self.exp_logger:
            train_log_metrics = {
                "loss": train_loss,
                "learning_rate": current_lr,
                **train_metrics,
            }
            self.exp_logger.log_metrics(train_log_metrics, step=epoch, split="train")

            val_log_metrics = {"loss": val_loss, **val_metrics}
            self.exp_logger.log_metrics(val_log_metrics, step=epoch, split="val")

    def _handle_checkpointing(self, epoch: int, val_acc: float) -> None:
        """Handle checkpoint saving logic."""
        # Save best model
        if val_acc > self.best_val_acc:
            logger.info(
                f"New best validation accuracy: {val_acc:.4f} "
                f"(prev: {self.best_val_acc:.4f})"
            )
            self.best_val_acc = val_acc
            self._save_checkpoint(epoch, is_best=True)

        # Save periodic checkpoint
        checkpoint_freq = getattr(self.config, 'checkpoint_freq', 1)
        if epoch % checkpoint_freq == 0 and epoch != self.config.training_params.train_epochs:
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a checkpoint using the checkpoint manager."""
        unwrapped_model = self._get_unwrapped_model()
        self.checkpoint_manager.save_checkpoint(
            model=unwrapped_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            best_val_acc=self.best_val_acc,
            is_best=is_best,
        )

    def _save_final_checkpoint(self) -> None:
        """Save final checkpoint."""
        unwrapped_model = self._get_unwrapped_model()
        self.checkpoint_manager.save_checkpoint(
            model=unwrapped_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self.config.training_params.train_epochs,
            best_val_acc=self.best_val_acc,
            is_final=True,
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint using the checkpoint manager."""
        unwrapped_model = self._get_unwrapped_model()
        state = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=unwrapped_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        self.start_epoch = state["start_epoch"]
        self.best_val_acc = state["best_val_acc"]

    def _broadcast_training_state(self) -> None:
        """Broadcast training state in distributed setting."""
        if not self.is_distributed:
            return

        import torch.distributed as dist

        # Create tensor for broadcasting
        state_tensor = torch.tensor(
            [float(self.start_epoch), self.best_val_acc],
            dtype=torch.float32,
            device=self.device,
        )

        # Broadcast from rank 0
        dist.broadcast(state_tensor, src=0)

        # Update state on non-main processes
        if not is_main_process():
            self.start_epoch = int(state_tensor[0].item())
            self.best_val_acc = state_tensor[1].item()
            logger.info(
                f"Rank {dist.get_rank()}: Synchronized training state - "
                f"start_epoch={self.start_epoch}, best_val_acc={self.best_val_acc:.4f}"
            )

    def _log_hyperparameters(self) -> None:
        """Log static hyperparameters."""
        if not is_main_process() or not self.exp_logger:
            return

        unwrapped_model = self._get_unwrapped_model()
        model_name = unwrapped_model.__class__.__name__

        self.exp_logger.log_params({
            "model_name": model_name,
            "epochs": self.config.training_params.train_epochs,
            "lr": self.config.training_params.lr,
            "batch_size": self.config.training_params.batch_size,
            "loss_fn": self.config.loss_function,
            "optimizer": self.optimizer.__class__.__name__,
            "weight_decay": self.config.training_params.weight_decay,
            "amp": self.amp_enabled,
            "amp_dtype": self.config.training_params.amp_dtype,
            "distributed": self.is_distributed,
            "world_size": self.world_size,
            "gradient_checkpointing": self.config.training_params.gradient_checkpointing,
            "scheduler": getattr(self.config.scheduler, "name", "none"),
            "warmup_steps": getattr(self.config.scheduler, "warmup_steps", 0),
            "min_lr": getattr(self.config.scheduler, "min_lr", 0),
            "checkpoint_freq": getattr(self.config, "checkpoint_freq", 1),
        })

    def _wrap_model_for_distributed(self, model: nn.Module) -> nn.Module:
        """Wrap model with DistributedDataParallel if needed."""
        if not self.is_distributed:
            return model

        logger.info(f"Wrapping model with DDP on rank {self.local_rank}")

        model_device = self.device
        if model_device.type == "cuda":
            # Use SLURM-aware device index
            device_index = get_local_device_index()
            logger.info(f"Using CUDA device index {device_index} for DDP")
            wrapped_model = parallel.DistributedDataParallel(
                model,
                device_ids=[device_index],
                output_device=device_index,
                find_unused_parameters=True,
            )
        else:
            # For CPU or when device index is None
            logger.info("Using CPU or auto-detect for DDP")
            wrapped_model = parallel.DistributedDataParallel(
                model,
                find_unused_parameters=True,
            )

        return wrapped_model

    def _get_unwrapped_model(self) -> nn.Module:
        """Get unwrapped model for checkpointing."""
        if isinstance(self.model, parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def _set_epoch_for_samplers(self, epoch: int) -> None:
        """Set epoch for distributed samplers."""
        if not self.is_distributed:
            return

        # Set epoch for distributed samplers
        if hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(epoch)
        if hasattr(self.eval_dataloader.sampler, "set_epoch"):
            self.eval_dataloader.sampler.set_epoch(epoch)
