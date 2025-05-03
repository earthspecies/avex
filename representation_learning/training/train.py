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
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from representation_learning.training.losses import ClipLoss, _build_criterion
from representation_learning.training.optimisers import _build_optimizer
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
    augmentation_processor : Optional[AugmentationProcessor], optional
        Processor for applying data augmentations, by default None
    exp_logger : Optional[ExperimentLogger], optional
        Experiment logger instance, by default None
    batch_size : int, optional
        Batch size (used for logging only), by default 32
    device : Optional[Union[str, torch.device]], optional
        Device to run training on, by default uses model device
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
        criterion: str = "cross_entropy",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        max_epochs: int = 10,
        amp: bool = False,
        amp_dtype: str = "bf16",  # one of: 'fp16', 'bf16'
        scheduler_config: Optional[Dict] = None,
        is_clip_mode: bool = False,
        checkpoint_freq: int = 1,
        augmentation_processor: Optional[AugmentationProcessor] = None,
        exp_logger: Optional[ExperimentLogger] = None,  # Make exp_logger optional
        batch_size: int = 32, # Add batch size to init
        device: Optional[Union[str, torch.device]] = None # Allow device to be passed
    ) -> None:
        self.model = model
        self.train_dataloader = train_dl
        self.eval_dataloader = eval_dl
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.is_clip_mode = is_clip_mode
        self.checkpoint_freq = checkpoint_freq
        self.max_epochs = max_epochs
        self.augmentation_processor = augmentation_processor
        self.log = exp_logger
        
        # Determine device
        if device is None:
             self.device = next(self.model.parameters()).device
        else:
             self.device = torch.device(device)
        self.model.to(self.device)

        # Distributed setup
        self.local_rank, self.world_size, self.is_distributed = setup_distributed()
        if self.is_distributed:
            # Ensure model is on correct device before wrapping
            self.model.to(self.device)
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
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_config or {},
            len(self.train_dataloader) * max_epochs,
        )

        # AMP
        self.amp_enabled = amp
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp_dtype]
        self.scaler = GradScaler() if self.amp_enabled and self.amp_dtype == torch.float16 else None

        # State
        self.best_val_acc: float = 0.0

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
                    "scheduler": scheduler_config.get('name', 'none') if scheduler_config else 'none',
                    "warmup_steps": scheduler_config.get('warmup_steps', 0) if scheduler_config else 0,
                    "min_lr": scheduler_config.get('min_lr', 0) if scheduler_config else 0,
                    "checkpoint_freq": self.checkpoint_freq,
                }
            )

    # ----------------------------- public API ------------------------------ #
    def train(self) -> None:
        """Run the full training loop for the configured number of epochs."""
        try:
            for epoch in range(1, self.max_epochs + 1):
                # Set epoch for distributed sampler
                if self.is_distributed:
                    # Check if sampler exists and has set_epoch method
                    if hasattr(self.train_dataloader.sampler, "set_epoch"):
                        self.train_dataloader.sampler.set_epoch(epoch)
                    if hasattr(self.eval_dataloader.sampler, "set_epoch"):
                         self.eval_dataloader.sampler.set_epoch(epoch) # Needed for consistent eval

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
                        logger.info(f"New best validation accuracy: {val_acc:.4f} (prev: {self.best_val_acc:.4f})")
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

        total_loss, total_correct, total_samples = 0.0, 0, 0
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"{'Train' if train else 'Eval '} Epoch {epoch}",
            leave=False,
            disable=not is_main_process(),  # Only show progress bar on master
        )

        grad_ctx = contextlib.nullcontext() if train else torch.no_grad()

        with grad_ctx:
            for step, batch in pbar:
                loss_tensor, correct, n = self._forward(batch, train=train)
                
                # Backward pass if training
                if train:
                    self._backward(loss_tensor)
                    
                # Accumulate metrics (ensure loss is detached and moved to CPU for aggregation)
                # Convert correct predictions count (int) to tensor for reduction
                batch_loss = loss_tensor.detach().item() * n
                total_loss += batch_loss
                total_correct += correct
                total_samples += n

                # Update progress bar (only on master)
                if is_main_process():
                    current_acc = total_correct / total_samples if total_samples > 0 else 0
                    # Display average loss per sample
                    avg_loss_so_far = total_loss / total_samples if total_samples > 0 else 0
                    pbar.set_postfix(
                        loss=f"{avg_loss_so_far:.4f}", 
                        acc=f"{current_acc:.4f}",
                    )

        # Reduce metrics across processes if distributed
        if self.is_distributed:
            metrics = torch.tensor([total_loss, total_correct, total_samples], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, total_correct, total_samples = metrics.tolist()
            total_correct = int(total_correct) # Convert back to int after reduction
            total_samples = int(total_samples)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        return avg_loss, avg_acc

    def _forward( 
        self,
        batch: Dict[str, Any], # Allow Any for flexibility with augmentations
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
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
        
        # Apply augmentations during training if available
        if train and self.augmentation_processor is not None:
            batch = self.augmentation_processor.apply_augmentations(batch)
        
        # Forward pass with optional AMP
        context_manager = autocast(enabled=self.amp_enabled, dtype=self.amp_dtype)
        with context_manager:
            if self.is_clip_mode:
                loss, correct, batch_size = self._forward_clip(batch)
            else:
                loss, correct, batch_size = self._forward_supervised(batch)

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
        padding_mask = batch.get("padding_mask") # Handle optional mask
        
        # Mixed labels from mixup augmentation if available
        mixed_target = batch.get("mixed_labels")
        
        # Forward pass
        outputs = self.model(audio, padding_mask=padding_mask)
        
        # Calculate loss - either with mixed labels or regular
        if mixed_target is not None:
            # For mixup - use soft labels (e.g., BCE or KLDiv)
            # Assuming outputs are logits, apply log_softmax for KLDiv or keep as is for BCE
            if isinstance(self.criterion, nn.CrossEntropyLoss):
                 log_probs = F.log_softmax(outputs, dim=1)
                 loss = F.kl_div(log_probs, mixed_target, reduction='batchmean')
            elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
                 loss = self.criterion(outputs, mixed_target)
            else:
                raise NotImplementedError(f"Mixup not implemented for criterion {type(self.criterion).__name__}")
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

    def _forward_clip(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, int, int]:
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
        
        # Ensure model has temperature attribute if needed by loss
        logit_scale = 1.0 # Default
        model_to_check = self.model_unwrapped
        if isinstance(self.criterion, ClipLoss) and hasattr(model_to_check, "temperature"):
            logit_scale = 1.0 / model_to_check.temperature
        
        audio_emb, text_emb = self.model(audio, text=text) # Pass text explicitly if model expects it

        # Get loss and logits from criterion
        if isinstance(self.criterion, ClipLoss):
            loss, logits = self.criterion(
                audio_emb, text_emb, logit_scale, output_logits=True
            )
        else:
            loss = self.criterion(audio_emb, text_emb, logit_scale)
            with torch.no_grad(): # Compute logits without grad if not returned by loss
                logits = audio_emb @ text_emb.T * logit_scale # Apply scale here if loss doesn't

        # For CLIP, accuracy is based on matching audio to correct text
        with torch.no_grad():
            ground_truth = torch.arange(len(audio), dtype=torch.long, device=self.device)
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
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, final: bool = False) -> None:
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
        
        save_model = self.model_unwrapped # Always save unwrapped model
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": save_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None, # Save scaler state
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
            return # Don't save if not periodic, best, or final
            
        model_path = self.model_dir / filename
        torch.save(checkpoint, model_path)
        logger.info(f"Saved checkpoint to {model_path}")
