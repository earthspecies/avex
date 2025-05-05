"""
representation_learning.training.train
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A compact yet full‑featured training loop that supports:

* Automatic Mixed Precision (fp16 / bf16)
* Gradient‑scaling when using fp16 AMP
* Proper no‑grad evaluation
* Check‑pointing of the best model
* Parameter & metric logging via `ExperimentLogger`
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from representation_learning.configs import RunConfig, EvaluateConfig
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
        self.model = model.to(device)
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

        # Log static hyper‑parameters once
        self.log.log_params(
            {
                "model_name": cfg.model_spec.name,
                "epochs": cfg.training_params.train_epochs,
                "lr": cfg.training_params.lr,
                "batch_size": cfg.training_params.batch_size,
                "loss_fn": cfg.loss_function,
                "amp": self.amp_enabled,
                "amp_dtype": cfg.training_params.amp_dtype,
            }
        )

        # Checkpoint directory
        self.ckpt_dir = Path(cfg.output_dir or "checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- public API ------------------------------ #
    def train(self) -> None:
        """Run the full training loop for the configured number of epochs."""
        for epoch in range(1, self.cfg.training_params.train_epochs + 1):
            print("running epoch")
            train_loss, train_acc = self._run_epoch(train=True, epoch=epoch)
            val_loss, val_acc = self._run_epoch(train=False, epoch=epoch)

            logger.info(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            # Log epoch‑level metrics
            self.log.log_metrics(
                {"loss": train_loss, "acc": train_acc}, step=epoch, split="train"
            )
            self.log.log_metrics(
                {"loss": val_loss, "acc": val_acc}, step=epoch, split="val"
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint("best.pt")

        self.log.finalize()

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
            loader, desc=f"{'Train' if train else 'Eval '} Epoch {epoch}", leave=False
        )

        grad_ctx = contextlib.nullcontext() if train else torch.no_grad()
        print("starting looping")

        with grad_ctx:
            for batch in pbar:
                loss, correct, n = self._forward(batch, train=train)
                total_loss += loss * n
                total_correct += correct
                total_samples += n
                pbar.set_postfix(loss=loss, acc=correct / n)

        return total_loss / total_samples, total_correct / total_samples

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
            Tuple of (loss value, number of correct predictions, batch size)
        """
        wav = batch["raw_wav"].to(self.device)
        mask = batch.get("padding_mask")
        if mask is not None:
            mask = mask.to(self.device)
        labels = batch["label"].to(self.device)

        # Forward (AMP works in both modes)
        with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            logits = (
                self.model(wav, padding_mask=mask)
                if mask is not None
                else self.model(wav)
            )
            loss = self.criterion(logits, labels)

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.amp_enabled and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        return loss.item(), correct, labels.size(0)

    def _save_checkpoint(self, name: str) -> None:
        ckpt_path = self.ckpt_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
            },
            ckpt_path,
        )
        logger.info("Saved checkpoint → %s", ckpt_path)

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
            multi_label: bool = False
        ):
            self.model = model
            self.optimizer = optimizer
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.device = device
            self.cfg = cfg
            self.log = exp_logger
            self.multi_label = multi_label
            
            # Set up loss function
            if self.multi_label:
                self.criterion = torch.nn.BCEWithLogitsLoss()
            else:
                self.criterion = torch.nn.CrossEntropyLoss()
            
            # Log static hyper-parameters
            self.log.log_params({
                "epochs": cfg.training_params.train_epochs,
                "lr": cfg.training_params.lr,
                "batch_size": cfg.training_params.batch_size,
                "loss_fn": "bce_with_logits" if self.multi_label else "cross_entropy",
            })
            
            self.best_val_acc = 0.0

        def train(self, num_epochs: int) -> None:
            """Run the full training loop for the configured number of epochs."""
            for epoch in range(1, num_epochs + 1):
                train_loss, train_acc = self._run_epoch(train=True, epoch=epoch)
                val_loss, val_acc = self._run_epoch(train=False, epoch=epoch)

                logger.info(
                    f"[Epoch {epoch:03d}] "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                )

                # Log epoch-level metrics
                self.log.log_metrics(
                    {"loss": train_loss, "acc": train_acc}, step=epoch, split="train"
                )
                self.log.log_metrics(
                    {"loss": val_loss, "acc": val_acc}, step=epoch, split="val"
                )

                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self._save_checkpoint("best.pt")

            self.log.finalize()

        def _run_epoch(self, train: bool, epoch: int) -> tuple[float, float]:
            """Run one epoch of training or validation."""
            loader = self.train_loader if train else self.val_loader

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch in tqdm(loader, desc=f"{'Train' if train else 'Eval '} Epoch {epoch}", leave=False):
                x = batch["raw_wav"].to(self.device)
                mask = batch.get("padding_mask")
                if mask is not None:
                    mask = mask.to(self.device)
                y = batch["label"].to(self.device)

                # Forward pass
                logits = (self.model(x, padding_mask=mask)
                    if mask is not None
                    else self.model(x)
                )
                loss = self.criterion(logits, y)

                # Backward pass if training
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Calculate accuracy
                if self.multi_label:
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    correct = (pred == y).all(dim=1).sum().item()
                else:
                    pred = logits.argmax(dim=1)
                    correct = (pred == y).sum().item()

                # Update metrics
                total_loss += loss.item() * y.size(0)
                total_correct += correct
                total_samples += y.size(0)

            return total_loss / total_samples, total_correct / total_samples

        def _save_checkpoint(self, name: str) -> None:
            """Save model checkpoint."""
            ckpt_path = Path(self.cfg.save_dir) / name
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_acc": self.best_val_acc,
                },
                ckpt_path,
            )
            logger.info("Saved checkpoint → %s", ckpt_path)