"""
representation_learning.training.train
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A compact training loop that supports:

* Automatic Mixed Precision (fp16 / bf16)
* Gradient‑scaling when using fp16 AMP
* Proper no‑grad evaluation
* Check‑pointing
* Parameter & metric logging via `ExperimentLogger`
* Distributed training with proper synchronization
* Learning rate scheduling with warmup

TODO: as things become more stable, remove or simplify benchmarking code.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.parallel as parallel
from esp_data.io.paths import GSPath, R2Path, anypath  # type: ignore
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from esp_data_temp.config import DatasetConfig
from representation_learning.configs import EvaluateConfig, RunConfig
from representation_learning.metrics.metric_factory import get_metric_class
from representation_learning.training.distributed import (
    cleanup_distributed,
    get_local_device_index,
    is_main_process,
)
from representation_learning.training.losses import ClipLoss, build_criterion
from representation_learning.training.training_utils import build_scheduler
from representation_learning.utils import ExperimentLogger

logger = logging.getLogger(__name__)


CloudPathT = GSPath | R2Path  # type: ignore[misc]


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
    is_eat_ssl : bool, optional
        Whether we're in EAT SSL training mode, by default False
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
    gradient_checkpointing : bool, optional
        Whether to enable gradient checkpointing, by default False
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
        is_eat_ssl: bool = False,
        checkpoint_freq: int = 1,
        exp_logger: Optional[ExperimentLogger] = None,
        batch_size: int = 32,
        device: Optional[Union[str, torch.device]] = None,
        resume_from_checkpoint: Optional[str] = None,
        run_config: Optional[RunConfig] = None,
        log_steps: int = 1,
        gradient_checkpointing: bool = False,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dl
        self.eval_dataloader = eval_dl
        self.model_dir = anypath(model_dir)

        # Ensure directory exists (for local; cloudpathlib handles remote lazily)
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except AttributeError:
            # Some CloudPath objects don't implement mkdir but create dirs on write
            pass
        self.is_clip_mode = is_clip_mode
        self.is_eat_ssl = is_eat_ssl
        self.checkpoint_freq = checkpoint_freq
        self.max_epochs = max_epochs
        self.log = exp_logger
        self.log_steps = log_steps

        # Timers for benchmarking (will be reset in _run_epoch and after each log
        # interval). Initialise with expected keys to avoid KeyError on first use.
        self.timers: Dict[str, float] = {
            "to_device": 0.0,
            "model_compute": 0.0,
        }

        # EAT-SSL per-component loss tracking
        # TODO: reminder to refactor the Trainer
        self._last_batch_ssl_metrics: Dict[str, float] | None = None

        # Determine device
        if device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for memory optimization")
            try:
                self.model.enable_gradient_checkpointing()
            except NotImplementedError as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")
            logger.info("gradient checkpointing enabled successfully")

        # Distributed setup - parameters are now passed in
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = is_distributed

        if self.is_distributed:
            logger.info(f"Wrapping model with DDP on rank {self.local_rank}")

            # Get the actual device index that the model is on
            model_device = self.device
            if model_device.type == "cuda":
                # Use SLURM-aware device index to handle GPU allocation properly
                device_index = get_local_device_index()
                logger.info(f"Using CUDA device index {device_index} for DDP")
                self.model = parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[device_index],
                    output_device=device_index,
                    find_unused_parameters=True,
                )
            else:
                # For CPU or when device index is None, let DDP auto-detect
                logger.info("Using CPU or auto-detect for DDP")
                self.model = parallel.DistributedDataParallel(
                    self.model,
                    find_unused_parameters=True,
                )
            self.model_unwrapped = self.model.module
        else:
            self.model_unwrapped = self.model

        # Optimizer, Criterion, Scheduler
        self.optimizer = optimizer
        self.criterion = build_criterion(criterion)
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

        # Broadcast critical state from main process to all processes in
        # distributed training
        if self.is_distributed:
            self._broadcast_training_state()

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
                    "gradient_checkpointing": gradient_checkpointing,
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

                # Capture detailed CLIP metrics from training epoch (if any)
                train_clip_metrics: Dict[str, float] | None = None
                if self.is_clip_mode and hasattr(self, "_last_epoch_clip_metrics"):
                    train_clip_metrics = self._last_epoch_clip_metrics.copy()

                val_loss, val_acc = self._run_epoch(train=False, epoch=epoch)

                # Log only on master process
                if is_main_process():
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if self.is_clip_mode and hasattr(self, "_last_epoch_clip_metrics"):
                        tr_a2t = self._last_epoch_clip_metrics.get("acc_a2t", 0.0)
                        tr_t2a = self._last_epoch_clip_metrics.get("acc_t2a", 0.0)

                    logger_line = (
                        f"[Epoch {epoch:03d}] "
                        f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                        f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
                        f"lr={current_lr:.2e}"
                    )

                    if (
                        self.is_clip_mode
                        and train_clip_metrics is not None
                        and hasattr(self, "_last_epoch_clip_metrics")
                    ):
                        tr_a2t = train_clip_metrics.get("acc_a2t", 0.0)
                        tr_t2a = train_clip_metrics.get("acc_t2a", 0.0)
                        val_a2t = self._last_epoch_clip_metrics.get("acc_a2t", 0.0)
                        val_t2a = self._last_epoch_clip_metrics.get("acc_t2a", 0.0)
                        logger_line += (
                            f" | train_a2t={tr_a2t:.4f} train_t2a={tr_t2a:.4f}"
                            f" val_a2t={val_a2t:.4f} val_t2a={val_t2a:.4f}"
                        )

                    logger.info(logger_line)

                    # Log epoch‑level metrics
                    if self.log:
                        train_metrics = {
                            "loss": train_loss,
                            "acc": train_acc,
                            "learning_rate": current_lr,
                        }
                        if train_clip_metrics:
                            train_metrics.update(train_clip_metrics)

                        self.log.log_metrics(
                            train_metrics,
                            step=epoch,
                            split="train",
                        )

                        val_metrics = {"loss": val_loss, "acc": val_acc}
                        if self.is_clip_mode and hasattr(
                            self, "_last_epoch_clip_metrics"
                        ):
                            val_metrics.update(self._last_epoch_clip_metrics)
                        self.log.log_metrics(val_metrics, step=epoch, split="val")

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
        finally:
            # Cleanup distributed training
            if self.is_distributed:
                cleanup_distributed()

            # Finalise experiment logging once training loop & cleanup are done.
            if is_main_process() and self.log:
                self.log.finalize()

    # --------------------------- internal helpers -------------------------- #
    def _run_epoch(self, *, train: bool, epoch: int) -> Tuple[float, float]:
        """
        Iterate once over the loader.

        Returns
        -------
        Tuple[float, float] : mean loss, mean accuracy
        """

        loader = self.train_dataloader if train else self.eval_dataloader

        if train:
            self.model.train()
        else:
            self.model.eval()

        # Reset per-epoch timers
        self.timers["to_device"] = 0.0
        self.timers["model_compute"] = 0.0

        total_loss: float = 0.0
        total_samples: int = 0

        # Separate correct counters for CLIP retrieval directions (a2t, t2a)
        total_correct: int = 0
        total_correct_a2t: int = 0
        total_correct_t2a: int = 0

        context_mgr = torch.enable_grad() if train else torch.no_grad()

        with context_mgr:
            for i, batch in enumerate(
                tqdm(
                    loader,
                    desc=f"{'Train' if train else 'Eval '} Epoch {epoch}",
                    leave=False,
                )
            ):
                # ------------------------------------------------------
                if not hasattr(self, "_global_updates"):
                    self._global_updates = 0  # type: ignore[attr-defined]

                if train:
                    self._global_updates += 1

                # ------------------------------------
                # Forward (and optional backward)
                # ------------------------------------
                loss, correct_out, batch_size = self._forward(batch, train=train)

                if train:
                    # Optimiser step (scaler if AMP-fp16)
                    self.optimizer.zero_grad()
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                    # FOR EAT-SSL: update the teacher
                    target_model = (
                        self.model_unwrapped if self.is_distributed else self.model
                    )
                    if self.is_eat_ssl:
                        backbone = target_model.backbone
                        backbone.set_num_updates(self._global_updates)

                    self.scheduler.step()

                # ------------------------------------
                # Accumulate statistics
                # ------------------------------------
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                if self.is_clip_mode:
                    # correct_out is Tuple[int,int] when CLIP mode
                    ca2t, ct2a = correct_out  # type: ignore[misc]
                    total_correct_a2t += ca2t
                    total_correct_t2a += ct2a
                else:
                    total_correct += correct_out  # type: ignore[misc]

                # Per-component EAT SSL losses --------------------------------
                if self.is_eat_ssl and self._last_batch_ssl_metrics is not None:
                    if "comp_totals" not in locals():  # lazily created once
                        comp_totals: Dict[str, float] = {}
                    for k, v in self._last_batch_ssl_metrics.items():
                        comp_totals[k] = comp_totals.get(k, 0.0) + v * batch_size

                # Per-log_steps logging
                if (i + 1) % self.log_steps == 0 or (i + 1) == len(loader):
                    avg_loss_so_far = (
                        total_loss / total_samples if total_samples else 0.0
                    )
                    if self.is_clip_mode:
                        avg_acc_a2t = (
                            total_correct_a2t / total_samples if total_samples else 0.0
                        )
                        avg_acc_t2a = (
                            total_correct_t2a / total_samples if total_samples else 0.0
                        )
                        avg_acc_so_far = (avg_acc_a2t + avg_acc_t2a) / 2.0
                    else:
                        avg_acc_so_far = (
                            total_correct / total_samples if total_samples else 0.0
                        )

                    # Build log line extensions for EAT SSL component losses
                    comp_log_str = ""
                    comp_log_metrics: Dict[str, float] = {}
                    if self.is_eat_ssl and "comp_totals" in locals():
                        for k, v in comp_totals.items():
                            comp_avg = v / total_samples
                            comp_log_str += f"  {k}={comp_avg:.2f}"
                            comp_log_metrics[k] = comp_avg

                    logger.info(
                        (
                            f"[LOG] Step {i + 1}/{len(loader)}: "
                            f"avg_loss={avg_loss_so_far:.4f}, "
                            f"avg_acc={avg_acc_so_far:.4f}"  # noqa: E501  (assembled)
                            f"{comp_log_str}"
                        )
                    )

                    if self.log is not None:
                        metrics_to_log = {
                            "loss": avg_loss_so_far,
                            "acc": avg_acc_so_far,
                            **comp_log_metrics,
                        }
                        self.log.log_metrics(metrics_to_log, step=self._global_updates)

        # ------------------------------------
        # Aggregate epoch metrics
        # ------------------------------------
        avg_loss = total_loss / total_samples if total_samples else 0.0

        if self.is_clip_mode:
            avg_acc_a2t = total_correct_a2t / total_samples if total_samples else 0.0
            avg_acc_t2a = total_correct_t2a / total_samples if total_samples else 0.0

            # Also log the current temperature (logit scale) for monitoring.
            if isinstance(self.model, parallel.DistributedDataParallel):
                clip_module = self.model.module  # type: ignore[attr-defined]
            else:
                clip_module = self.model  # type: ignore[assignment]

            current_scale = (
                clip_module.logit_scale.exp().item()  # type: ignore[attr-defined]
                if hasattr(clip_module, "logit_scale")
                else 1.0
            )

            self._last_epoch_clip_metrics = {
                "acc_a2t": avg_acc_a2t,
                "acc_t2a": avg_acc_t2a,
                "logit_scale": current_scale,
            }

            avg_acc = (avg_acc_a2t + avg_acc_t2a) / 2.0
        else:
            avg_acc = total_correct / total_samples if total_samples else 0.0

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
                loss, correct_pair, batch_size = self._forward_clip(batch)
                correct = correct_pair  # can be tuple
            elif self.is_eat_ssl:
                loss, correct, batch_size = self._forward_eat_ssl(batch)
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

        # Forward pass
        outputs = self.model(audio, padding_mask=padding_mask)

        # ------------------------------------------------------------------
        #  Match target format to criterion expectations
        # ------------------------------------------------------------------
        if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
            # BCE expects float multi-hot targets (shape [B, C]).  If the
            # dataset provides integer class indices, convert them to one-hot.
            if target.dim() == 1:
                target = torch.nn.functional.one_hot(
                    target.long(), num_classes=outputs.size(1)
                ).float()
        elif isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            # Cross-entropy expects class indices (shape [B]).  If the dataset
            # (Collater) already produced one-hot vectors (shape [B, C]),
            # collapse to indices.
            if target.dim() > 1:
                target = target.argmax(dim=1)

        loss = self.criterion(outputs, target)

        # --------------------------------------------------
        # Accuracy calculation (single- vs multi-label)
        # --------------------------------------------------
        with torch.no_grad():
            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                # Multi-label: exact-match accuracy after thresholding.
                prob = torch.sigmoid(outputs)
                pred = (prob > 0.5).float()
                if target.dtype != torch.float32:
                    target = target.float()
                correct = (pred.eq(target).all(dim=1)).sum().item()
            else:  # Cross-entropy (single-label)
                _, predicted = outputs.max(1)
                # target is guaranteed to be indices after the conversion above
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

        # Forward pass through CLIPModel – now returns embeddings *and* logit scale
        audio_emb, text_emb, logit_scale = self.model(
            audio, text=text, padding_mask=padding_mask
        )

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

        # Accuracy metrics
        #   • audio→text
        #   • text →audio

        with torch.no_grad():
            local_bs = audio.size(0)

            rank = 0
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()

            start = rank * local_bs
            end = start + local_bs

            local_logits = logits[start:end, start:end]

            ground_truth = torch.arange(local_bs, device=self.device, dtype=torch.long)

            # audio → text ---------------------------------------------------
            pred_a2t = torch.argmax(local_logits, dim=1)
            correct_a2t = (pred_a2t == ground_truth).sum().item()

            # text → audio ---------------------------------------------------
            pred_t2a = torch.argmax(local_logits, dim=0)
            correct_t2a = (pred_t2a == ground_truth).sum().item()

        return loss, (correct_a2t, correct_t2a), local_bs

    def _forward_eat_ssl(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, int, int]:
        """Forward path for self-supervised EAT pre-training.

        The EAT backbone returns ``{"losses": {...}, "sample_size": int}``. We
        aggregate the component losses and divide by *sample_size* so the final
        scalar is invariant to clip length.

        Returns
        -------
        tuple[torch.Tensor, int, int]
            ``(loss, 0, batch_size)`` – accuracy is undefined in SSL so zero is
            returned for the *correct* count.

        Raises
        ------
        RuntimeError
            If the backbone fails to return the expected dictionary structure.
        """

        audio = batch["raw_wav"]
        padding_mask = batch.get("padding_mask")

        out = self.model(audio, padding_mask=padding_mask)

        if not isinstance(out, dict) or "losses" not in out:
            raise RuntimeError(
                "EAT model did not return expected loss dict in SSL mode"
            )

        # Per-component averages (before weighting by batch size).  Store
        # them so the calling loop can accumulate and log them.
        self._last_batch_ssl_metrics = {
            k: v.sum().item() / out["sample_size"].clamp(min=1).item()
            for k, v in out["losses"].items()
        }

        # Add additional EAT metrics that are already computed
        if "masked_pct" in out and not type(out["masked_pct"]) == float:
            self._last_batch_ssl_metrics["masked_pct"] = out["masked_pct"].item()

        if "target_var" in out and not type(out["target_var"]) == float:
            self._last_batch_ssl_metrics["target_var"] = out["target_var"].item()

        # Add prediction variance metrics (there can be multiple pred_var_i)
        for key, value in out.items():
            if key.startswith("pred_var_"):
                self._last_batch_ssl_metrics[key] = value.item()

        total_loss = sum(v.sum() for v in out["losses"].values())
        sample_size = out.get("sample_size", audio.size(0)).clamp(min=1).float()
        loss = total_loss / sample_size

        # accuracy is undefined in SSL; return zero so logging remains intact
        return loss, 0, audio.size(0)

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
        """Save a checkpoint of the model, optimizer, and training state."""
        if not is_main_process():
            return  # Only the main process saves checkpoints

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

        # Decide where to place the checkpoint:
        #   • If user provided a cloud path → always use that.
        #   • Otherwise fall back to ExperimentLogger.log_dir when available.
        if isinstance(self.model_dir, CloudPathT):  # type: ignore[arg-type]
            ckpt_dir = self.model_dir
        elif self.log is not None and hasattr(self.log, "log_dir"):
            ckpt_dir = Path(self.log.log_dir)
        else:
            ckpt_dir = Path(self.model_dir)

        # Make sure directory exists (local) or is implicitly handled (cloud).
        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        except AttributeError:
            pass  # Cloud paths may not implement mkdir

        ckpt_path = ckpt_dir / filename

        # torch.save requires a writable file-like object or a local path. To
        # support cloud paths we use the .open('wb') API when dealing with
        # cloudpathlib objects.
        if isinstance(ckpt_path, CloudPathT):  # type: ignore[arg-type]
            with ckpt_path.open("wb") as f:
                torch.save(checkpoint, f)
        else:
            torch.save(checkpoint, ckpt_path)

        logger.info("Saved checkpoint → %s", ckpt_path)

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
                checkpoint["model_state_dict"],
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

    def _broadcast_training_state(self) -> None:
        """Broadcast critical training state from main process to all processes."""
        if not self.is_distributed:
            return

        import torch.distributed as dist

        # Create tensors for the state we need to broadcast
        # Using float tensors since dist.broadcast requires tensors
        state_tensor = torch.tensor(
            [
                float(self.start_epoch),
                self.best_val_acc,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Broadcast from rank 0 to all other ranks
        dist.broadcast(state_tensor, src=0)

        # Update state on non-main processes
        if not is_main_process():
            self.start_epoch = int(state_tensor[0].item())
            self.best_val_acc = state_tensor[1].item()
            logger.info(
                f"Rank {dist.get_rank()}: Synchronized training state - "
                f"start_epoch={self.start_epoch}, best_val_acc={self.best_val_acc:.4f}"
            )


class FineTuneTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        cfg: EvaluateConfig,
        exp_logger: ExperimentLogger,
        dataset_cfg: DatasetConfig,
        num_classes: int,
        multi_label: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.log = exp_logger
        self.multi_label = multi_label
        self.dataset_cfg = dataset_cfg
        self.num_classes = num_classes
        
        # Initialize metric configuration from dataset config
        self.metric_names = dataset_cfg.metrics
        self.primary_metric_name = self.metric_names[0]  # Use first metric as primary

        # Set up loss function
        if self.multi_label:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Log static hyper-parameters
        self.log.log_params(
            {
                "epochs": cfg.training_params.train_epochs,
                "lr": cfg.training_params.lr,
                "batch_size": cfg.training_params.batch_size,
                "loss_fn": "bce_with_logits" if self.multi_label else "cross_entropy",
            }
        )

        self.best_val_metric = 0.0
        # Track best epoch metrics for return value
        self.best_train_metrics: dict[str, float] = {"loss": float("inf")}
        self.best_val_metrics: dict[str, float] = {"loss": float("inf")}

    def train(self, num_epochs: int) -> tuple[dict[str, float], dict[str, float]]:
        """
        Run the full fine-tuning loop and return best train/val metrics.

        Returns
        -------
        tuple[dict[str, float], dict[str, float]]
            A tuple of (best_train_metrics, best_val_metrics) collected across epochs.
        """
        for epoch in range(1, num_epochs + 1):
            train_loss, train_metrics = self._run_epoch(train=True, epoch=epoch)
            val_loss, val_metrics = self._run_epoch(train=False, epoch=epoch)

            # Get primary metric values
            train_primary = train_metrics[self.primary_metric_name]
            val_primary = val_metrics[self.primary_metric_name]

            # Log epoch progress
            logger.info(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f}  train_{self.primary_metric_name}={train_primary:.4f} | "
                f"val_loss={val_loss:.4f}  val_{self.primary_metric_name}={val_primary:.4f}"
            )

            # Log epoch-level metrics
            train_log_metrics = {"loss": train_loss}
            train_log_metrics.update(train_metrics)
            val_log_metrics = {"loss": val_loss}
            val_log_metrics.update(val_metrics)
            
            self.log.log_metrics(train_log_metrics, step=epoch, split="train")
            self.log.log_metrics(val_log_metrics, step=epoch, split="val")

            # Save best model based on primary metric
            if val_primary > self.best_val_metric:
                self.best_val_metric = val_primary
                self._save_checkpoint("best.pt")
                # Update best metrics
                self.best_train_metrics = {"loss": train_loss}
                self.best_train_metrics.update(train_metrics)
                self.best_val_metrics = {"loss": val_loss}
                self.best_val_metrics.update(val_metrics)

        # Load the best model after training is complete
        self._load_best_checkpoint()

        return self.best_train_metrics, self.best_val_metrics

    def _run_epoch(self, train: bool, epoch: int) -> tuple[float, dict[str, float]]:
        """Run one epoch of training or validation.

        Parameters
        ----------
        train : bool
            If ``True`` performs a training step; otherwise runs evaluation.
        epoch : int
            Current epoch number (1-indexed).

        Returns
        -------
        tuple[float, dict[str, float]]
            A tuple ``(loss, metrics_dict)`` computed over the entire epoch.
        """
        loader = self.train_loader if train else self.val_loader
        
        # Create fresh metric instances for this epoch
        metrics = [get_metric_class(m, self.num_classes) for m in self.metric_names]

        total_loss = 0.0
        total_samples = 0

        # Set model mode
        if train:
            self.model.train()
        else:
            self.model.eval()

        for batch in tqdm(
            loader, desc=f"{'Train' if train else 'Eval '} Epoch {epoch}", leave=False
        ):
            if "embed" in batch:
                z = batch["embed"].to(self.device)
                logits = self.model(z)
                y = batch["label"].to(self.device)
            else:
                x = batch["raw_wav"].to(self.device)
                mask = batch.get("padding_mask")
                if mask is not None:
                    mask = mask.to(self.device)
                y = batch["label"].to(self.device)

                logits = (
                    self.model(x, padding_mask=mask)
                    if mask is not None
                    else self.model(x)
                )

            loss = self.criterion(logits, y)

            # Backward pass if training
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update metrics with current batch
            for metric in metrics:
                metric.update(logits, y)

            # Update loss tracking
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

        # Compute final metrics for this epoch
        epoch_metrics = {
            name: metric.get_primary_metric()
            for name, metric in zip(self.metric_names, metrics, strict=False)
        }

        return total_loss / total_samples, epoch_metrics

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        # Save inside the experiment-specific folder to avoid collisions across
        # datasets / experiments that share the same EvaluateConfig.save_dir
        if self.log is not None and hasattr(self.log, "log_dir"):
            ckpt_dir = Path(self.log.log_dir)
        else:
            ckpt_dir = Path(self.cfg.save_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_metric": self.best_val_metric,
            },
            ckpt_path,
        )
        logger.info("Saved checkpoint → %s", ckpt_path)

    def _load_best_checkpoint(self) -> None:
        """Load the best model checkpoint from disk."""
        # Get the path to the best checkpoint
        if self.log is not None and hasattr(self.log, "log_dir"):
            ckpt_dir = Path(self.log.log_dir)
        else:
            ckpt_dir = Path(self.cfg.save_dir)
        ckpt_path = ckpt_dir / "best.pt"

        if not ckpt_path.exists():
            logger.warning(
                f"Best checkpoint not found at {ckpt_path}. Using current model state."
            )
            return

        # Load the checkpoint
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Restored best model from {ckpt_path} (val_{self.primary_metric_name}: {self.best_val_metric:.4f})"
        )
