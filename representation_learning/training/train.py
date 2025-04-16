"""
representation_learning.training.train
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

from representation_learning.utils import ExperimentLogger


def _build_criterion(name: str) -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name in {"bce", "binary_cross_entropy_with_logits"}:
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss_function: {name}")


class Trainer:
    """
    Parameters
    ----------
    model         : nn.Module
    optimizer     : torch.optim.Optimizer
    train_loader  : DataLoader
    val_loader    : DataLoader
    device        : torch.device
    cfg           : RunConfig (or any object with the referenced fields)
    exp_logger    : object with `.log_params`, `.log_metrics`, `.finalize`
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: Any,
        exp_logger: ExperimentLogger,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.log = exp_logger

        # Setup AMP
        self.amp_enabled = cfg.training_params.amp
        self.amp_dtype = torch.bfloat16 if cfg.training_params.amp_dtype == "bf16" else torch.float16
        self.scaler = GradScaler() if self.amp_dtype == torch.float16 else None

        self.criterion = _build_criterion(cfg.loss_function)
        self.best_val_acc: float = 0.0

        # Log static hyper‑parameters once
        self.log.log_params(
            {
                "model_name": cfg.model_name,
                "epochs": cfg.training_params.train_epochs,
                "lr": cfg.training_params.lr,
                "batch_size": cfg.training_params.batch_size,
                "loss_fn": cfg.loss_function,
                "amp": self.amp_enabled,
                "amp_dtype": cfg.training_params.amp_dtype,
            }
        )

        # Checkpoint directory
        self.ckpt_dir = Path("checkpoints")
        self.ckpt_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def train(self, num_epochs: int) -> None:
        for epoch in range(1, num_epochs + 1):
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

    # --------------------------------------------------------------------- #
    #  Internal helpers
    # --------------------------------------------------------------------- #
    def _run_epoch(self, *, train: bool, epoch: int) -> Tuple[float, float]:
        loader = self.train_loader if train else self.val_loader
        mode = "train" if train else "eval"
        self.model.train(mode == "train")

        total_loss, total_correct, total_samples = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"{mode.title()} Epoch {epoch}", leave=False)

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
        wav = batch["raw_wav"].to(self.device)
        mask = batch.get("padding_mask")
        if mask is not None:
            mask = mask.to(self.device)
        labels = batch["label"].to(self.device)

        # Forward pass with AMP
        with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            logits = (
                self.model(wav, padding_mask=mask)
                if mask is not None
                else self.model(wav)
            )
            loss = self.criterion(logits, labels)

        if train:
            self.optimizer.zero_grad()
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
