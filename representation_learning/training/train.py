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
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.parallel as parallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from representation_learning.configs import RunConfig
from representation_learning.data.dataset import AugmentationProcessor
from representation_learning.training.distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    reduce_dict,
    setup_distributed,
)
from representation_learning.training.losses import _build_criterion
from representation_learning.training.optimisers import _build_optimizer, _build_scheduler
from representation_learning.utils import ExperimentLogger  # type: ignore

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
    optimizer     : torch.optim.Optimizer
    train_loader  : DataLoader
    val_loader    : DataLoader
    device        : torch.device
    cfg           : RunConfig (or any object with the referenced fields)
    exp_logger    : ExperimentLogger
    checkpoint_freq : int, optional
        How often to save checkpoints (in epochs), defaults to 1 (once per epoch)
    """

    # ----------------------------- initialisation -------------------------- #
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: RunConfig,
        exp_logger: ExperimentLogger,
        checkpoint_freq: int = 1,
        augmentation_processor: Optional[AugmentationProcessor] = None,
    ) -> None:
        # Setup distributed training if needed
        self.local_rank, self.world_size, _ = setup_distributed(
            backend=cfg.distributed_backend,
            port=cfg.distributed_port,
        )
        self.is_distributed = self.local_rank is not None

        # Move model to device and wrap with DDP if needed
        self.model = model.to(device)
        if self.is_distributed:
            self.model = parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.log = exp_logger
        self.checkpoint_freq = checkpoint_freq

        # AMP
        self.amp_enabled: bool = cfg.training_params.amp
        self.amp_dtype = (
            torch.bfloat16 if cfg.training_params.amp_dtype == "bf16" else torch.float16
        )
        self.scaler: GradScaler | None = (
            GradScaler()
            if (self.amp_enabled and self.amp_dtype == torch.float16)
            else None
        )

        # Setup learning rate scheduler
        total_steps = len(train_loader) * cfg.training_params.train_epochs
        self.scheduler = build_scheduler(optimizer, cfg, total_steps)

        self.criterion = _build_criterion(cfg.loss_function)
        self.best_val_acc: float = 0.0

        # Log static hyper‑parameters once (only on master process)
        if is_main_process():
            self.log.log_params(
                {
                    "model_name": cfg.model_spec.name,
                    "epochs": cfg.training_params.train_epochs,
                    "lr": cfg.training_params.lr,
                    "batch_size": cfg.training_params.batch_size,
                    "loss_fn": cfg.loss_function,
                    "amp": self.amp_enabled,
                    "amp_dtype": cfg.training_params.amp_dtype,
                    "distributed": self.is_distributed,
                    "world_size": self.world_size,
                    "scheduler": cfg.scheduler.name,
                    "warmup_steps": cfg.scheduler.warmup_steps,
                    "min_lr": cfg.scheduler.min_lr,
                    "checkpoint_freq": self.checkpoint_freq,
                }
            )

            # Checkpoint directory
            self.ckpt_dir = Path(cfg.output_dir or "checkpoints")
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Augmentations
        self.augmentation_processor = augmentation_processor

    # ----------------------------- public API ------------------------------ #
    def train(self) -> None:
        """Run the full training loop for the configured number of epochs."""
        try:
            for epoch in range(1, self.cfg.training_params.train_epochs + 1):
                # Set epoch for distributed sampler
                if self.is_distributed:
                    self.train_loader.sampler.set_epoch(epoch)

                train_loss, train_acc = self._run_epoch(train=True, epoch=epoch)
                val_loss, val_acc = self._run_epoch(train=False, epoch=epoch)

                # Step the scheduler
                self.scheduler.step()

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
                        self.best_val_acc = val_acc
                        self._save_checkpoint("best.pt")

                    # Save periodic checkpoint based on frequency
                    if epoch % self.checkpoint_freq == 0:
                        self._save_checkpoint(f"checkpoint_epoch_{epoch:03d}.pt")

            # Save final checkpoint
            if is_main_process():
                self._save_checkpoint("final.pt")
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
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)  # train=True -> .train(); train=False -> .eval()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        pbar = tqdm(
            loader,
            desc=f"{'Train' if train else 'Eval '} Epoch {epoch}",
            leave=False,
            disable=not is_main_process(),  # Only show progress bar on master
        )

        grad_ctx = contextlib.nullcontext() if train else torch.no_grad()

        with grad_ctx:
            for batch in pbar:
                loss, correct, n = self._forward(batch, train=train)
                total_loss += loss * n
                total_correct += correct
                total_samples += n

                # Update progress bar (only on master)
                if is_main_process():
                    pbar.set_postfix(loss=loss, acc=correct / n)

        # Reduce metrics across processes
        metrics = {
            "loss": torch.tensor(total_loss, device=self.device),
            "correct": torch.tensor(total_correct, device=self.device),
            "samples": torch.tensor(total_samples, device=self.device),
        }
        reduced_metrics = reduce_dict(metrics)

        return (
            reduced_metrics["loss"].item() / reduced_metrics["samples"].item(),
            reduced_metrics["correct"].item() / reduced_metrics["samples"].item(),
        )

    def _forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        train: bool,
    ) -> Tuple[float, int, int]:
        """Single forward/backward step (if `train=True`).

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Input batch containing raw_wav, padding_mask (optional), and label
        train : bool
            Whether this is a training step (affects gradient computation)

        Returns
        -------
        Tuple[float, int, int]
            (loss, number of correct predictions, batch size)
        """
        # Move batch to device
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()
        }

        # Apply augmentations during training if available
        if train and self.augmentation_processor is not None:
            batch = self.augmentation_processor.apply_augmentations(batch)

        # Forward pass with optional AMP
        with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            if self._is_clip_mode():
                loss, correct, batch_size = self._forward_clip(batch)
            else:
                logits = self.model(batch["raw_wav"], batch["padding_mask"])
                loss = self.criterion(logits, batch["label"])
                pred = logits.argmax(dim=-1)
                correct = (pred == batch["label"]).sum().item()
                batch_size = len(batch["label"])

        if train:
            # Backward pass with optional gradient scaling
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item(), correct, batch_size

    def _is_clip_mode(self) -> bool:
        """Check if the current model/loss setup is CLIP-style contrastive learning.

        Returns
        -------
        bool
            True if using CLIP-style contrastive learning
        """
        return (
            isinstance(self.criterion, nn.Module)
            and self.criterion.__class__.__name__.lower() == "cliploss"
        )

    def _forward_clip(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, int, int]:
        """Forward and loss computation for CLIPModel + ClipLoss.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Input batch containing raw_wav and text_label

        Returns
        -------
        Tuple[torch.Tensor, int, int]
            (loss, number of correct predictions, batch size)
        """
        audio = batch["raw_wav"]
        text = batch["text_label"]
        audio_emb, text_emb = self.model(audio, text)
        logit_scale = (
            1.0 / self.model.temperature if hasattr(self.model, "temperature") else 1.0
        )
        # Get loss and logits from criterion
        loss, logits = self.criterion(
            audio_emb, text_emb, logit_scale, output_logits=True
        )
        labels = torch.arange(logits.size(0), device=logits.device)
        pred = logits.argmax(dim=-1)
        correct = (pred == labels).sum().item()
        batch_size = logits.size(0)
        return loss, correct, batch_size

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        name : str
            Name of the checkpoint file
        """
        if not is_main_process():
            return

        # Get state dict from DDP model if needed
        if self.is_distributed:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        torch.save(
            {
                "model": state_dict,
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.cfg.training_params.train_epochs,
                "best_val_acc": self.best_val_acc,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            },
            self.ckpt_dir / name,
        )
        logger.info(f"Saved checkpoint: {self.ckpt_dir / name}")
