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
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.parallel as parallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from representation_learning.configs import RunConfig
from representation_learning.training.distributed import (
    cleanup_distributed,
    is_master_process,
    reduce_dict,
    setup_distributed,
)
from representation_learning.utils import ExperimentLogger  # type: ignore

logger = logging.getLogger(__name__)


def _build_criterion(name: str) -> nn.Module:
    """Factory for common loss functions.

    Parameters
    ----------
    name : str
        Name of the loss function to create

    Returns
    -------
    nn.Module
        The requested loss function module

    Raises
    ------
    ValueError
        If the loss function name is unknown
    """
    name = name.lower()
    if name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss()
    if name in {"bce", "binary_cross_entropy_with_logits", "bce_with_logits"}:
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss_function: {name}")


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

        self.criterion = _build_criterion(cfg.loss_function)
        self.best_val_acc: float = 0.0

        # Log static hyper‑parameters once (only on master process)
        if is_master_process():
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
                }
            )

            # Checkpoint directory
            self.ckpt_dir = Path(cfg.output_dir or "checkpoints")
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

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

                # Log only on master process
                if is_master_process():
                    logger.info(
                        f"[Epoch {epoch:03d}] "
                        f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                        f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                    )

                    # Log epoch‑level metrics
                    self.log.log_metrics(
                        {"loss": train_loss, "acc": train_acc},
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

            if is_master_process():
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
            disable=not is_master_process(),  # Only show progress bar on master
        )

        grad_ctx = contextlib.nullcontext() if train else torch.no_grad()

        with grad_ctx:
            for batch in pbar:
                loss, correct, n = self._forward(batch, train=train)
                total_loss += loss * n
                total_correct += correct
                total_samples += n

                # Update progress bar (only on master)
                if is_master_process():
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
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass with optional AMP
        with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            logits = self.model(batch["raw_wav"])
            loss = self.criterion(logits, batch["label"])

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

        # Compute accuracy
        pred = logits.argmax(dim=-1)
        correct = (pred == batch["label"]).sum().item()
        return loss.item(), correct, len(batch["label"])

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        name : str
            Name of the checkpoint file
        """
        if not is_master_process():
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
            },
            self.ckpt_dir / name,
        )
