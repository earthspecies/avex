"""
representation_learning.training.train
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A compact yet full‑featured training loop that supports:

* Automatic Mixed Precision (fp16 / bf16)
* Gradient‑scaling when using fp16 AMP
* Proper no‑grad evaluation
* Check‑pointing of the best model
* Parameter & metric logging via `ExperimentLogger`
* Distributed training with proper synchronization
* Learning rate scheduling with warmup
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as parallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from representation_learning.configs import RunConfig
from representation_learning.training.distributed import (
    cleanup_distributed,
    is_main_process,
)
from representation_learning.training.losses import ClipLoss, _build_criterion
from representation_learning.training.training_utils import build_scheduler
from representation_learning.utils import ExperimentLogger

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """
    Generic supervised trainer.

    Parameters
    ----------
    model         : nn.Module
        Network to train
    optimizer     : torch.optim.Optimizer
        Optimizer instance
    train_dl  : DataLoader
        Training data loader
    eval_dl    : DataLoader
        Validation data loader
    model_dir : Union[str, Path]
        Directory to save checkpoints
    criterion : str, optional
        Loss function name, by default "cross_entropy"
    lr : float, optional
        Learning rate, by default 3e-4
    weight_decay : float, optional
        Weight decay, by default 0.0
    max_epochs : int, optional
        Maximum number of epochs, by default 10
    amp : bool, optional
        Whether to use automatic mixed precision, by default False
    amp_dtype : str, optional
        AMP data type, one of: 'fp16', 'bf16', by default "bf16"
    scheduler_config : Optional[Dict], optional
        LR scheduler config, by default None
    is_clip_mode : bool, optional
        Whether we're in CLIP training mode, by default False
    checkpoint_freq : int, optional
        Frequency of checkpointing (in epochs), defaults to 1 (once per epoch)
    exp_logger : Optional[ExperimentLogger], optional
        Experiment logger instance, by default None
    batch_size : int, optional
        Batch size (used for logging only), by default 32
    device : Optional[Union[str, torch.device]], optional
        Device to run training on, by default uses model device
    resume_from_checkpoint : Optional[str], optional
        Path to checkpoint to resume from, by default None
    run_config : Optional[RunConfig], optional
        Configuration for learning rate scheduler, by default None
    local_rank : int
        Local rank for distributed training
    world_size : int
        World size for distributed training
    is_distributed : bool
        Whether distributed training is enabled
    log_steps : int, optional
        Frequency of logging benchmarking results, by default 100
    """

    # ----------------------------- initialisation -------------------------- #
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dl: DataLoader,
        eval_dl: DataLoader,
        model_dir: Union[str, Path],
        local_rank: int,
        world_size: int,
        is_distributed: bool,
        criterion: str = "cross_entropy",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        max_epochs: int = 10,
        amp: bool = False,
        amp_dtype: str = "bf16",  # one of: 'fp16', 'bf16'
        scheduler_config: Optional[Dict] = None,
        is_clip_mode: bool = False,
        checkpoint_freq: int = 1,
        exp_logger: Optional[ExperimentLogger] = None,
        batch_size: int = 32,
        device: Optional[Union[str, torch.device]] = None,
        resume_from_checkpoint: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        log_steps: int = 1,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dl
        self.eval_dataloader = eval_dl
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.is_clip_mode = is_clip_mode
        self.checkpoint_freq = checkpoint_freq
        self.max_epochs = max_epochs
        self.log = exp_logger
        self.log_steps = log_steps

        # Timers for benchmarking (will be reset in _run_epoch and after each log
        # interval)
        self.timers: Dict[str, float] = {}

        # Determine device
        if device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

        # Distributed setup - parameters are now passed in
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = is_distributed

        if self.is_distributed:
            logger.info(f"Wrapping model with DDP on rank {self.local_rank}")
            self.model = parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
            self.model_unwrapped = self.model.module
        else:
            self.model_unwrapped = self.model

        # Optimizer, Criterion, Scheduler
        self.optimizer = optimizer
        self.criterion = _build_criterion(criterion)
        self.scheduler = (
            build_scheduler(
                self.optimizer,
                run_config,  # Pass the full config object
                len(self.train_dataloader) * max_epochs,
            )
            if run_config
            else None
        )  # Handle case where config isn't passed

        # AMP
        self.amp_enabled = amp
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp_dtype]
        self.scaler = (
            GradScaler()
            if self.amp_enabled and self.amp_dtype == torch.float16
            else None
        )

        # State
        self.best_val_acc: float = 0.0
        self.start_epoch: int = 1  # Default start epoch

        # Load checkpoint if specified (only on main process)
        if is_main_process() and resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)

        # Log static hyper‑parameters once (only on master process)
        if is_main_process() and self.log:
            try:
                # Attempt to get model name from unwrapped model
                model_name = self.model_unwrapped.__class__.__name__
            except AttributeError:
                model_name = "Unknown"
            self.log.log_params(
                {
                    "model_name": model_name,
                    "epochs": max_epochs,
                    "lr": lr,
                    "batch_size": batch_size,
                    "loss_fn": criterion,
                    "optimizer": optimizer.__class__.__name__,
                    "weight_decay": weight_decay,
                    "amp": self.amp_enabled,
                    "amp_dtype": amp_dtype,
                    "distributed": self.is_distributed,
                    "world_size": self.world_size,
                    "scheduler": scheduler_config.get("name", "none")
                    if scheduler_config
                    else "none",
                    "warmup_steps": scheduler_config.get("warmup_steps", 0)
                    if scheduler_config
                    else 0,
                    "min_lr": scheduler_config.get("min_lr", 0)
                    if scheduler_config
                    else 0,
                    "checkpoint_freq": self.checkpoint_freq,
                }
            )

    # ----------------------------- public API ------------------------------ #
    def train(self) -> None:
        """Run the full training loop for the configured number of epochs."""
        try:
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                # Set epoch for distributed sampler
                if self.is_distributed:
                    # Check if sampler exists and has set_epoch method
                    if hasattr(self.train_dataloader.sampler, "set_epoch"):
                        self.train_dataloader.sampler.set_epoch(epoch)
                    if hasattr(self.eval_dataloader.sampler, "set_epoch"):
                        self.eval_dataloader.sampler.set_epoch(
                            epoch
                        )  # Needed for consistent eval

                train_loss, train_acc = self._run_epoch(train=True, epoch=epoch)
                val_loss, val_acc = self._run_epoch(train=False, epoch=epoch)

                # Log only on master process
                if is_main_process():
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"[Epoch {epoch:03d}] "
                        f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                        f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
                        f"lr={current_lr:.2e}"
                    )

                    # Log epoch‑level metrics
                    if self.log:
                        self.log.log_metrics(
                            {
                                "loss": train_loss,
                                "acc": train_acc,
                                "learning_rate": current_lr,
                            },
                            step=epoch,
                            split="train",
                        )
                        self.log.log_metrics(
                            {"loss": val_loss, "acc": val_acc}, step=epoch, split="val"
                        )

                    # Save best model
                    if val_acc > self.best_val_acc:
                        logger.info(
                            f"New best validation accuracy: {val_acc:.4f} "
                            f"(prev: {self.best_val_acc:.4f})"
                        )
                        self.best_val_acc = val_acc
                        self._save_checkpoint(epoch, is_best=True)

                    # Save periodic checkpoint based on frequency
                    if epoch % self.checkpoint_freq == 0 and epoch != self.max_epochs:
                        self._save_checkpoint(epoch)

            # Save final checkpoint
            if is_main_process():
                self._save_checkpoint(self.max_epochs, final=True)
                if self.log:
                    self.log.finalize()
        finally:
            # Cleanup distributed training
            if self.is_distributed:
                cleanup_distributed()

    # --------------------------- internal helpers -------------------------- #
    def _run_epoch(self, *, train: bool, epoch: int) -> Tuple[float, float]:
        """
        Iterate once over the loader.

        Returns
        -------
        Tuple[float, float] : mean loss, mean accuracy
        """
        loader = self.train_dataloader if train else self.eval_dataloader
        self.model.train(train)  # train=True -> .train(); train=False -> .eval()

        # Initialize/Reset timers and counters for the current epoch and log intervals
        self.timers = {"to_device": 0.0, "model_compute": 0.0}
        interval_dataloader_time = 0.0
        interval_backward_time = 0.0
        steps_since_last_log = 0

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        last_log_time_marker = time.perf_counter()  # For wall time of log interval

        iter_data_wait_start_time = time.perf_counter()  # For individual data load time

        total_loss, total_correct, total_samples = 0.0, 0, 0
        pbar_desc = f"{'Train' if train else 'Eval '} Epoch {epoch}"
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=pbar_desc,
            leave=False,
            disable=not is_main_process(),
        )

        grad_ctx = contextlib.nullcontext() if train else torch.no_grad()

        with grad_ctx:
            for batch_idx, batch in pbar:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                iter_data_ready_time = time.perf_counter()
                current_iter_dataloader_time = (
                    iter_data_ready_time - iter_data_wait_start_time
                )
                interval_dataloader_time += current_iter_dataloader_time
                steps_since_last_log += 1

                loss_tensor, correct, n = self._forward(batch, train=train)

                current_iter_backward_time = 0.0
                if train:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    bwd_pass_start_time = time.perf_counter()
                    self._backward(loss_tensor)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    current_iter_backward_time = (
                        time.perf_counter() - bwd_pass_start_time
                    )
                    interval_backward_time += current_iter_backward_time

                batch_loss = loss_tensor.detach().item() * n
                total_loss += batch_loss
                total_correct += correct
                total_samples += n

                if is_main_process():
                    current_acc = (
                        total_correct / total_samples if total_samples > 0 else 0
                    )
                    avg_loss_so_far = (
                        total_loss / total_samples if total_samples > 0 else 0
                    )
                    pbar.set_postfix(
                        loss=f"{avg_loss_so_far:.4f}", acc=f"{current_acc:.4f}"
                    )

                # Log benchmark times every self.log_steps
                if is_main_process() and steps_since_last_log == self.log_steps:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    current_time_marker = time.perf_counter()
                    interval_wall_time = current_time_marker - last_log_time_marker

                    logger.info(
                        f"--- Benchmarking Results for last {steps_since_last_log} "
                        f"steps ("
                    )
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx + 1}, "
                        f"{'Train' if train else 'Eval'}) ---"
                    )
                    logger.info(f"Wall Time for interval: {interval_wall_time:.4f}s")

                    component_times = {
                        "Dataloading": interval_dataloader_time,
                        "Data-to-Device": self.timers["to_device"],
                        "Model Forward Compute": self.timers["model_compute"],
                        "Model Backward": interval_backward_time if train else 0.0,
                    }

                    total_accounted_time = sum(component_times.values())
                    other_time = interval_wall_time - total_accounted_time

                    if (
                        interval_wall_time > 1e-6
                    ):  # Avoid division by zero if interval is too short
                        dataloading_s = component_times["Dataloading"]
                        dataloading_pct = (dataloading_s / interval_wall_time) * 100
                        data_to_device_s = component_times["Data-to-Device"]
                        data_to_device_pct = (
                            data_to_device_s / interval_wall_time
                        ) * 100
                        model_forward_s = component_times["Model Forward Compute"]
                        model_forward_pct = (model_forward_s / interval_wall_time) * 100
                        model_backward_s = component_times["Model Backward"]
                        model_backward_pct = (
                            model_backward_s / interval_wall_time
                        ) * 100
                        logger.info(
                            f"  {'Dataloading:':<25} "
                            f"{dataloading_s:.4f}s "
                            f"({dataloading_pct:.2f}%)"
                        )
                        logger.info(
                            f"  {'Data-to-Device:':<25} "
                            f"{data_to_device_s:.4f}s "
                            f"({data_to_device_pct:.2f}%)"
                        )
                        logger.info(
                            f"  {'Model Forward Compute:':<25} "
                            f"{model_forward_s:.4f}s "
                            f"({model_forward_pct:.2f}%)"
                        )
                        if train:
                            logger.info(
                                f"  {'Model Backward:':<25} "
                                f"{model_backward_s:.4f}s "
                                f"({model_backward_pct:.2f}%)"
                            )
                        other_time_pct = (other_time / interval_wall_time) * 100
                        logger.info(
                            f"  {'Other/Overhead:':<25} "
                            f"{other_time:.4f}s "
                            f"({other_time_pct:.2f}%)"
                        )
                    else:
                        logger.info(
                            "Interval wall time too short for percentage breakdown."
                        )
                    logger.info("--- End Benchmarking Results ---")

                    # Reset timers for the next interval
                    self.timers = {"to_device": 0.0, "model_compute": 0.0}
                    interval_dataloader_time = 0.0
                    interval_backward_time = 0.0
                    steps_since_last_log = 0
                    if (
                        self.device.type == "cuda"
                    ):  # Ensure resets are captured before new timing
                        torch.cuda.synchronize(self.device)
                    last_log_time_marker = time.perf_counter()

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                iter_data_wait_start_time = time.perf_counter()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        # Reduce metrics across processes if distributed
        if self.is_distributed:
            metrics = torch.tensor(
                [total_loss, total_correct, total_samples], device=self.device
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, total_correct, total_samples = metrics.tolist()
            total_correct = int(total_correct)  # Convert back to int after reduction
            total_samples = int(total_samples)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        return avg_loss, avg_acc

    def _forward(
        self,
        batch: Dict[str, Any],  # Allow Any for flexibility with augmentations
        *,
        train: bool,
    ) -> Tuple[torch.Tensor, int, int]:
        """Single forward pass (backward is handled in _run_epoch).

        Parameters
        ----------
        batch : Dict[str, Any]
            Input batch
        train : bool
            Whether this is a training step

        Returns
        -------
        Tuple[torch.Tensor, int, int]
            (loss tensor, number of correct predictions, batch size)
        """
        # Move batch to device
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_start_to_device = time.perf_counter()
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timers["to_device"] += time.perf_counter() - t_start_to_device

        # Forward pass with optional AMP
        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )  # Ensure prev ops (like to_device) are done
        t_start_model_compute = time.perf_counter()
        context_manager = autocast(enabled=self.amp_enabled, dtype=self.amp_dtype)
        with context_manager:
            if self.is_clip_mode:
                loss, correct, batch_size = self._forward_clip(batch)
            else:
                loss, correct, batch_size = self._forward_supervised(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timers["model_compute"] += time.perf_counter() - t_start_model_compute

        return loss, correct, batch_size

    def _forward_supervised(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, int, int]:
        """Forward and loss computation for standard supervised learning.

        Parameters
        ----------
        batch : Dict[str, Any]
            Input batch containing raw_wav and label

        Returns
        -------
        Tuple[torch.Tensor, int, int]
            (loss tensor, number of correct predictions, batch size)
        """
        # Get the inputs
        audio = batch["raw_wav"]
        target = batch["label"]
        padding_mask = batch.get("padding_mask")  # Handle optional mask

        # Mixed labels from mixup augmentation if available
        mixed_target = batch.get("mixed_labels")

        # Forward pass
        outputs = self.model(audio, padding_mask=padding_mask)

        # Calculate loss - either with mixed labels or regular
        if mixed_target is not None:
            # For mixup - use soft labels (e.g., BCE or KLDiv)
            # Assuming outputs are logits, apply log_softmax for KLDiv
            # or keep as is for BCE
            if isinstance(self.criterion, nn.CrossEntropyLoss):
                log_probs = F.log_softmax(outputs, dim=1)
                loss = F.kl_div(log_probs, mixed_target, reduction="batchmean")
            elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
                loss = self.criterion(outputs, mixed_target)
            else:
                raise NotImplementedError(
                    f"Mixup not implemented for criterion "
                    f"{type(self.criterion).__name__}"
                )
        else:
            loss = self.criterion(outputs, target)

        # Calculate accuracy
        with torch.no_grad():
            if mixed_target is not None:
                _, predicted = outputs.max(1)
                _, true_targets = mixed_target.max(1)
                correct = (predicted == true_targets).sum().item()
            else:
                _, predicted = outputs.max(1)
                correct = (predicted == target).sum().item()

        return loss, correct, target.size(0)

    def _forward_clip(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, int, int]:
        """Forward and loss computation for CLIPModel + ClipLoss.

        Parameters
        ----------
        batch : Dict[str, Any]
            Input batch containing raw_wav and text_label

        Returns
        -------
        Tuple[torch.Tensor, int, int]
            (loss tensor, number of correct predictions, batch size)
        """
        audio = batch["raw_wav"]
        text = batch["text_label"]
        padding_mask = batch.get("padding_mask")

        # Ensure model has temperature attribute if needed by loss
        logit_scale = 1.0  # Default
        model_to_check = self.model_unwrapped
        if isinstance(self.criterion, ClipLoss) and hasattr(
            model_to_check, "temperature"
        ):
            logit_scale = 1.0 / model_to_check.temperature

        audio_emb, text_emb = self.model(
            audio, text=text, padding_mask=padding_mask
        )  # Pass text explicitly if model expects it

        # Get loss and logits from criterion
        if isinstance(self.criterion, ClipLoss):
            loss, logits = self.criterion(
                audio_emb, text_emb, logit_scale, output_logits=True
            )
        else:
            loss = self.criterion(audio_emb, text_emb, logit_scale)
            with torch.no_grad():  # Compute logits without grad if not returned by loss
                logits = (
                    audio_emb @ text_emb.T * logit_scale
                )  # Apply scale here if loss doesn't

        # For CLIP, accuracy is based on matching audio to correct text
        with torch.no_grad():
            ground_truth = torch.arange(
                len(audio), dtype=torch.long, device=self.device
            )
            pred_indices = torch.argmax(logits, dim=1)
            correct = (pred_indices == ground_truth).sum().item()

        batch_size = audio.size(0)
        return loss, correct, batch_size

    def _backward(self, loss: torch.Tensor) -> None:
        """Backwards step with AMP support."""
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:  # fp16
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:  # bf16 or full precision
            loss.backward()
            self.optimizer.step()

        # Step the scheduler after the optimizer step
        if self.scheduler is not None:
            self.scheduler.step()

    def _save_checkpoint(
        self, epoch: int, is_best: bool = False, final: bool = False
    ) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch
        is_best : bool, optional
            Whether this is the best model so far, by default False
        final : bool, optional
            Whether this is the final model checkpoint, by default False
        """
        if not is_main_process():
            return

        save_model = self.model_unwrapped  # Always save unwrapped model

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": save_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "scaler_state_dict": self.scaler.state_dict()
            if self.scaler
            else None,  # Save scaler state
            "best_val_acc": self.best_val_acc,
        }

        # Determine filename
        if final:
            filename = "final_model.pt"
        elif is_best:
            filename = "best_model.pt"
        elif epoch % self.checkpoint_freq == 0:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"
        else:
            return  # Don't save if not periodic, best, or final

        model_path = self.model_dir / filename
        torch.save(checkpoint, model_path)
        logger.info(f"Saved checkpoint to {model_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model, optimizer, scheduler, and scaler state from a checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(
                f"Checkpoint file not found: {checkpoint_path}. "
                f"Starting training from scratch."
            )
            return

        # Load checkpoint on the same device it was saved to avoid issues, then map
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load model state
        try:
            # Handle potential missing keys and unexpected keys
            self.model_unwrapped.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
            logger.info(f"Loaded model state from {checkpoint_path}")
        except Exception as e:
            logger.error(
                f"Error loading model state_dict: {e}. "
                f"Model weights might be incompatible."
            )
            return  # Stop loading if model is incompatible

        # Load optimizer state
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Loaded optimizer state.")
        except Exception as e:
            logger.warning(
                f"Could not load optimizer state: {e}. "
                f"Optimizer will start from scratch."
            )

        # Load scheduler state
        if (
            self.scheduler
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"]
        ):
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Loaded scheduler state.")
            except Exception as e:
                logger.warning(
                    f"Could not load scheduler state: {e}. "
                    f"Scheduler will start from scratch."
                )

        # Load scaler state
        if (
            self.scaler
            and "scaler_state_dict" in checkpoint
            and checkpoint["scaler_state_dict"]
        ):
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                logger.info("Loaded AMP scaler state.")
            except Exception as e:
                logger.warning(
                    f"Could not load AMP scaler state: {e}. "
                    f"Scaler will start from scratch."
                )

        # Load training state
        self.start_epoch = checkpoint.get("epoch", 0) + 1  # Start from next epoch
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)

        logger.info(
            f"Resuming training from epoch {self.start_epoch} with best validation "
            f"accuracy {self.best_val_acc:.4f}"
        )
