"""Fine-tuning utilities for representation learning models.

This module provides utilities for fine-tuning models during evaluation,
including probe training and model optimization.
"""

import logging
import multiprocessing
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim.lr_scheduler as lr_scheduler
from esp_data.io import anypath, exists
from tqdm import tqdm

from representation_learning.configs import EvaluateConfig, ExperimentConfig
from representation_learning.metrics.metric_factory import get_metric_class
from representation_learning.models.probes.utils.factory import build_probe_from_config
from representation_learning.training.optimisers import get_optimizer
from representation_learning.utils import ExperimentLogger, universal_torch_load

logger = logging.getLogger("run_finetune")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


# -------------------------------------------------------------------- #
#  FineTuneTrainer
# -------------------------------------------------------------------- #
class FineTuneTrainer:
    """Fine-tuning trainer for representation learning models.

    Handles the training loop for fine-tuning models during evaluation,
    including probe training and optimization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        cfg: EvaluateConfig,
        exp_logger: ExperimentLogger,
        num_labels: int,
        multi_label: bool = False,
        dataset_metrics: Optional[List[str]] = None,
        warmup_epochs: int = 5,
        scheduler_type: str = "cosine",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.log = exp_logger
        self.num_labels = num_labels
        self.multi_label = multi_label
        self.dataset_metrics = dataset_metrics
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler_type
        self.base_lr = cfg.training_params.lr

        # Set up loss function
        if self.multi_label:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Set up learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Log static hyper-parameters
        log_params = {
            "epochs": cfg.training_params.train_epochs,
            "lr": cfg.training_params.lr,
            "batch_size": cfg.training_params.batch_size,
            "loss_fn": ("bce_with_logits" if self.multi_label else "cross_entropy"),
        }

        # Add gradient clipping parameter if specified
        if hasattr(cfg.training_params, "gradient_clip_val") and cfg.training_params.gradient_clip_val is not None:
            log_params["gradient_clip_val"] = cfg.training_params.gradient_clip_val

        # Add scheduler parameters
        log_params["warmup_epochs"] = self.warmup_epochs
        log_params["scheduler_type"] = self.scheduler_type

        self.log.log_params(log_params)

        # Determine primary metric name based on task type and dataset configuration
        if self.dataset_metrics and len(self.dataset_metrics) > 0:
            self.primary_metric_name = self.dataset_metrics[0].lower()
        else:
            # For detection tasks (multi_label=True), require explicit metrics
            if self.multi_label:
                raise ValueError("No dataset metrics provided for multi-label task. ")
            else:
                # For single-label classification, still allow accuracy as default
                self.primary_metric_name = "accuracy"

        self.best_val_metric = 0.0
        # Track best epoch metrics for return value
        self.best_train_metrics: dict[str, float] = {
            "loss": float("inf"),
            self.primary_metric_name: 0.0,
        }
        self.best_val_metrics: dict[str, float] = {
            "loss": float("inf"),
            self.primary_metric_name: 0.0,
        }

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration.

        Returns
        -------
        Optional[torch.optim.lr_scheduler._LRScheduler]
            The learning rate scheduler or None if scheduler_type is "none".
        """
        if self.scheduler_type == "none":
            return None

        # Calculate total steps for cosine annealing
        total_epochs = self.cfg.training_params.train_epochs
        steps_per_epoch = len(self.train_loader)
        total_steps = total_epochs * steps_per_epoch

        if self.scheduler_type == "cosine":
            # Cosine annealing with warmup
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - (self.warmup_epochs * steps_per_epoch),
                eta_min=self.base_lr * 0.01,  # Minimum LR is 1% of base LR
            )
        elif self.scheduler_type == "linear":
            # Linear decay with warmup
            return lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,  # End at 1% of base LR
                total_iters=total_steps - (self.warmup_epochs * steps_per_epoch),
            )
        elif self.scheduler_type == "step":
            # Step decay
            return lr_scheduler.StepLR(
                self.optimizer,
                step_size=total_epochs // 3,  # Decay every 1/3 of total epochs
                gamma=0.5,
            )
        else:
            logger.warning(f"Unknown scheduler type: {self.scheduler_type}. Using no scheduler.")
            return None

    def _get_warmup_lr(self, epoch: int) -> float:
        """Calculate learning rate during warmup period.

        Returns
        -------
        float
            The learning rate for the given epoch during warmup.
        """
        if epoch <= self.warmup_epochs:
            # Linear warmup from 0 to base_lr
            return self.base_lr * (epoch / self.warmup_epochs)
        else:
            return self.base_lr

    def train(self, num_epochs: int) -> tuple[dict[str, float], dict[str, float]]:
        """
        Run the full fine-tuning loop and return best train/val metrics.

        Returns
        -------
        tuple[dict[str, float], dict[str, float]]
            A tuple of (best_train_metrics, best_val_metrics) collected across epochs.
        """
        for epoch in range(1, num_epochs + 1):
            # Start timing the epoch
            epoch_start_time = time.time()

            # Set current epoch for scheduler stepping
            self._current_epoch = epoch

            # Apply learning rate warmup only if scheduler is enabled
            if self.scheduler is not None and epoch <= self.warmup_epochs:
                warmup_lr = self._get_warmup_lr(epoch)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            train_loss, train_metric = self._run_epoch(train=True, epoch=epoch)
            val_loss, val_metric = self._run_epoch(train=False, epoch=epoch)

            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time

            # Get current learning rate for logging
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f}  "
                f"train_{self.primary_metric_name}={train_metric:.4f} | "
                f"val_loss={val_loss:.4f}  "
                f"val_{self.primary_metric_name}={val_metric:.4f} | "
                f"lr={current_lr:.6f} | "
                f"epoch_duration={epoch_duration:.2f}s"
            )

            # Log epoch-level metrics
            self.log.log_metrics(
                {
                    "loss": train_loss,
                    self.primary_metric_name: train_metric,
                    "epoch_duration": epoch_duration,
                },
                step=epoch,
                split="train",
            )
            self.log.log_metrics(
                {
                    "loss": val_loss,
                    self.primary_metric_name: val_metric,
                    "epoch_duration": epoch_duration,
                },
                step=epoch,
                split="val",
            )

            # Save best model (higher is better for all our metrics)
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self._save_checkpoint("best.pt")
                # Update best metrics
                self.best_train_metrics = {
                    "loss": train_loss,
                    self.primary_metric_name: train_metric,
                }
                self.best_val_metrics = {
                    "loss": val_loss,
                    self.primary_metric_name: val_metric,
                }

        # Load the best model after training is complete
        self._load_best_checkpoint()

        return self.best_train_metrics, self.best_val_metrics

    def _run_epoch(self, train: bool, epoch: int) -> tuple[float, float]:
        """Run one epoch of training or validation.

        Parameters
        ----------
        train : bool
            If ``True`` performs a training step; otherwise runs evaluation.
        epoch : int
            Current epoch number (1-indexed).

        Returns
        -------
        tuple[float, float]
            A tuple ``(loss, metric_value)`` computed over the entire epoch.

        Raises
        ------
        RuntimeError
            If metric initialization fails due to configuration issues
        """
        loader = self.train_loader if train else self.val_loader

        total_loss = 0.0
        total_samples = 0

        try:
            metric_calculator = get_metric_class(self.primary_metric_name, self.num_labels)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize metric '{self.primary_metric_name}' "
                f"for {self.num_labels} classes. "
                f"Error: {e}. This indicates a configuration issue - ensure "
                f"the metric name is valid "
                f"and properly supported by the metric factory."
            ) from e

        # Use tqdm only if not disabled in config
        if not self.cfg.disable_tqdm:
            iterator = tqdm(
                loader,
                desc=f"{'Train' if train else 'Eval '} Epoch {epoch}",
                leave=False,
            )
        else:
            iterator = loader

        for batch in iterator:
            if "embed" in batch:
                # Single tensor case (backward compatibility)
                z = batch["embed"].to(self.device)
                logits = self.model(z)
                y = batch["label"].to(self.device)
            elif "raw_wav" in batch:
                # Raw audio case
                x = batch["raw_wav"].to(self.device)
                mask = batch.get("padding_mask")
                if mask is not None:
                    mask = mask.to(self.device)
                y = batch["label"].to(self.device)

                logits = self.model(x, padding_mask=mask) if mask is not None else self.model(x)
            else:
                # Dictionary embeddings case (no raw_wav key)
                embed_keys = [k for k in batch.keys() if k != "label"]
                z = {}
                for k in embed_keys:
                    value = batch[k]
                    if isinstance(value, list):
                        # Filter out Nones safely
                        tensors = [t for t in value if t is not None]
                        z[k] = [t.to(self.device) for t in tensors]
                    else:
                        if value is None:
                            continue
                        z[k] = value.to(self.device)
                logits = self.model(z)
                y = batch["label"].to(self.device)

            # Calculate loss - handle different label formats for different loss
            # functions
            if self.multi_label:
                # BCEWithLogitsLoss expects one-hot encoded labels
                loss = self.criterion(logits, y)
            else:
                # CrossEntropyLoss expects class indices, not one-hot encoded labels
                y_indices = y.argmax(dim=1)
                loss = self.criterion(logits, y_indices)

            # Backward pass if training
            if train:
                self.optimizer.zero_grad()
                loss.backward()

                # Apply gradient clipping if specified in config
                if (
                    hasattr(self.cfg.training_params, "gradient_clip_val")
                    and self.cfg.training_params.gradient_clip_val is not None
                ):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training_params.gradient_clip_val,
                    )

                self.optimizer.step()

                # Step scheduler after optimizer step (for proper LR scheduling)
                # Only step scheduler if it exists and we're past the warmup period
                if (
                    self.scheduler is not None
                    and hasattr(self, "_current_epoch")
                    and self._current_epoch > self.warmup_epochs
                ):
                    self.scheduler.step()

            metric_calculator.update(logits, y)

            # Update loss
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

        metric_value = metric_calculator.get_primary_metric()

        return total_loss / total_samples, metric_value

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
                "primary_metric_name": self.primary_metric_name,
            },
            ckpt_path,
        )
        logger.info("Saved checkpoint → %s", ckpt_path)

    def _load_best_checkpoint(self) -> None:
        """Load the best model checkpoint from disk."""
        # Get the path to the best checkpoint
        if self.log is not None and hasattr(self.log, "log_dir"):
            ckpt_dir = anypath(self.log.log_dir)
        else:
            ckpt_dir = anypath(self.cfg.save_dir)
        ckpt_path = ckpt_dir / "best.pt"

        if not exists(ckpt_path):
            logger.warning(f"Best checkpoint not found at {ckpt_path}. Using current model state.")
            return

        # Load the checkpoint
        checkpoint = universal_torch_load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Re-register hooks on base_model if needed (they can be cleared by load)
        try:
            if hasattr(self.model, "base_model") and hasattr(self.model, "layers"):
                base_model = self.model.base_model
                layers = self.model.layers
                if (
                    base_model is not None
                    and hasattr(base_model, "register_hooks_for_layers")
                    and hasattr(base_model, "_hooks")
                    and isinstance(base_model._hooks, dict)
                    and len(base_model._hooks) == 0
                ):
                    logging.getLogger("run_finetune").info("Re-registering hooks on base_model for layers: %s", layers)
                    layers = base_model.register_hooks_for_layers(layers)
        except Exception as hook_err:
            logging.getLogger("run_finetune").warning(
                "Could not re-register base_model hooks after checkpoint load: %s",
                hook_err,
            )
        logger.info(
            f"Restored best model from {ckpt_path} (val_{self.primary_metric_name}: {self.best_val_metric:.4f})"
        )


# -------------------------------------------------------------------- #
#  Linear-probe helper
# -------------------------------------------------------------------- #
def train_and_eval_offline(
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    input_dim: List[Tuple[int, ...]],
    num_labels: int,
    layer_names: List[str],
    eval_cfg: EvaluateConfig,
    device: torch.device,
    exp_logger: ExperimentLogger,
    multi_label: bool,
    dataset_metrics: Optional[List[str]] = None,
    experiment_cfg: Optional[ExperimentConfig] = None,
    target_length: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Train a linear probe and evaluate it on *cached* test embeddings.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
        • **train_metrics** – aggregated over training split
        • **val_metrics**   – aggregated over validation split
        • **probe_test_metrics** – metrics on cached-embedding test split

    Raises
    ------
    ValueError
        If no dataset metrics are provided in the evaluation configuration
    """

    # Validate embeddings_dims
    if input_dim is None or len(input_dim) == 0:
        raise ValueError(
            "input_dim is required for offline training but was not provided. "
            "This usually happens when loading embeddings from H5 files. "
            "Please ensure embedding dimensions are properly extracted from the "
            "dataset."
        )

    # Use experiment's probe configuration if available, otherwise create default
    if experiment_cfg and experiment_cfg.probe_config:
        probe_config = experiment_cfg.probe_config
        logger.info("Using experiment probe configuration")
    else:
        # Require explicit probe configuration for consistent behavior
        raise ValueError(
            "Probe configuration is required. Please provide experiment_cfg with "
            "probe_config to ensure proper probe settings and behavior."
        )

    logger.info(
        f"Creating offline probe: type={probe_config.probe_type}, input_dim={input_dim}, target_length={target_length}"
    )

    probe = build_probe_from_config(
        probe_config=probe_config,
        base_model=None,
        num_classes=num_labels,
        device=device,
        feature_mode=True,
        input_dim=input_dim,
        target_length=target_length,
    )
    # Count parameters for offline probe
    total_params = sum(p.numel() for p in probe.parameters())
    trainable_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)

    def format_params(count: int, total: int) -> str:
        if count >= 1e9:
            return f"{count / 1e9:.2f}B trainable / {total / 1e9:.2f}B total"
        elif count >= 1e6:
            return f"{count / 1e6:.2f}M trainable / {total / 1e6:.2f}M total"
        elif count >= 1e3:
            return f"{count / 1e3:.2f}K trainable / {total / 1e3:.2f}K total"
        else:
            return f"{count} trainable / {total} total"

    logger.info(f"Offline probe → {format_params(trainable_params, total_params)}")
    probe.train()

    optim = get_optimizer(probe.parameters(), eval_cfg.training_params)

    # Store probe model in exp_logger for later access (e.g., printing learned weights)
    exp_logger.probe_model = probe

    # Use spawn context for DataLoaders to avoid fork-related issues (e.g., HDF5)
    ctx = multiprocessing.get_context("spawn")

    trainer = FineTuneTrainer(
        model=probe,
        optimizer=optim,
        train_loader=torch.utils.data.DataLoader(
            train_ds,
            batch_size=eval_cfg.training_params.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=eval_cfg.num_workers,
            multiprocessing_context=ctx,
        ),
        val_loader=torch.utils.data.DataLoader(
            val_ds,
            batch_size=eval_cfg.training_params.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=eval_cfg.num_workers,
            multiprocessing_context=ctx,
        ),
        device=device,
        cfg=eval_cfg,
        exp_logger=exp_logger,
        num_labels=num_labels,
        multi_label=multi_label,
        dataset_metrics=dataset_metrics,
        warmup_epochs=5,  # 5 epochs warmup
        scheduler_type=eval_cfg.training_params.scheduler_type,  # Use scheduler type
        # from config
    )

    train_metrics, val_metrics = trainer.train(num_epochs=eval_cfg.training_params.train_epochs)

    # ---------- probe evaluation on cached test embeddings ----------
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=eval_cfg.training_params.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=eval_cfg.num_workers,
        multiprocessing_context=ctx,
    )

    # Metric selection - require explicit metrics, no fallbacks
    if dataset_metrics is not None and len(dataset_metrics) > 0:
        metric_names = dataset_metrics
    else:
        raise ValueError("Expected metrics to be specified in the evaluation configuration. ")

    metrics = [get_metric_class(m, num_labels) for m in metric_names]

    probe.eval()  # feature_mode stays True (inputs are embeddings)
    with torch.no_grad():
        for batch in test_loader:
            # Handle both dictionary and single tensor cases
            if "embed" in batch:
                # Single tensor case (backward compatibility)
                z = batch["embed"].to(device)
            else:
                # Dictionary case: pass all embedding keys to probe
                embed_keys = [k for k in batch.keys() if k != "label"]
                if not embed_keys:
                    raise ValueError("No embedding keys found in batch")

                # Create dictionary with all embedding layers
                z = {}
                for k in embed_keys:
                    if isinstance(batch[k], list):
                        # Handle list of tensors (aggregation="none" case)
                        z[k] = [tensor.to(device) for tensor in batch[k]]
                    else:
                        # Handle single tensor (other aggregation methods)
                        z[k] = batch[k].to(device)
                logger.debug(f"Using all layers for test evaluation: {embed_keys}")

            y = batch["label"].to(device)

            logits = probe(z)
            for met in metrics:
                met.update(logits, y)

    probe_test_metrics = {name: met.get_primary_metric() for name, met in zip(metric_names, metrics, strict=False)}
    return train_metrics, val_metrics, probe_test_metrics


# -------------------------------------------------------------------- #
#  Full fine-tune helper
# -------------------------------------------------------------------- #
def train_and_eval_online(
    train_dl_raw: torch.utils.data.DataLoader,
    val_dl_raw: torch.utils.data.DataLoader,
    test_dl_raw: torch.utils.data.DataLoader,
    base_model: torch.nn.Module,
    num_labels: int,
    layer_names: List[str],
    eval_cfg: EvaluateConfig,
    device: torch.device,
    exp_logger: ExperimentLogger,
    multi_label: bool,
    dataset_metrics: Optional[List[str]] = None,
    experiment_cfg: Optional[ExperimentConfig] = None,
    target_length: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Train a model by fine-tuning on raw waveforms.

    Parameters
    ----------
    train_dl_raw : torch.utils.data.DataLoader
        Training dataloader with raw waveforms
    val_dl_raw : torch.utils.data.DataLoader
        Validation dataloader with raw waveforms
    test_dl_raw : torch.utils.data.DataLoader
        Test dataloader with raw waveforms
    base_model : torch.nn.Module
        Base model to fine-tune
    num_labels : int
        Number of output classes
    layer_names : List[str]
        Names of layers to extract features from
    eval_cfg : EvaluateConfig
        Evaluation configuration
    device : torch.device
        Device to train on
    exp_logger : ExperimentLogger
        Logger for experiment metrics
    multi_label : bool
        Whether this is a multi-label classification task
    dataset_metrics : Optional[List[str]]
        List of metrics to compute for this dataset
    experiment_cfg : Optional[ExperimentConfig]
        Experiment configuration containing probe settings
    target_length : Optional[int]
        Target length in samples for audio processing

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
        • **train_metrics** – aggregated over training split
        • **val_metrics**   – aggregated over validation split
        • **probe_test_metrics** – metrics on test split

    Raises
    ------
    ValueError
        If no dataset metrics are provided in the evaluation configuration
    """
    # Use experiment's probe configuration if available, otherwise create default
    if experiment_cfg and experiment_cfg.probe_config:
        probe_config = experiment_cfg.probe_config
        logger.info("Using experiment probe configuration for online training")
    else:
        # Require explicit probe configuration for online training
        raise ValueError(
            "Online training requires experiment configuration with probe_config. "
            "No fallback to default configuration is allowed to ensure proper "
            "probe behavior and settings."
        )

    logger.info(
        f"Creating online training model: type={probe_config.probe_type}, "
        f"frozen={probe_config.freeze_backbone}, target_length={target_length}"
    )

    sft_model = build_probe_from_config(
        probe_config=probe_config,
        base_model=base_model,
        num_classes=num_labels,
        device=device,
        feature_mode=False,
        input_dim=None,
        frozen=probe_config.freeze_backbone,
        target_length=target_length,
    )
    # Count parameters separately for probe and base model
    total_params = sum(p.numel() for p in sft_model.parameters())
    trainable_params = sum(p.numel() for p in sft_model.parameters() if p.requires_grad)

    # Count probe vs base model parameters
    probe_total = 0
    probe_trainable = 0
    base_total = 0
    base_trainable = 0

    for name, param in sft_model.named_parameters():
        if hasattr(sft_model, "base_model") and sft_model.base_model is not None:
            # Check if this parameter belongs to the base model
            is_base_param = any(
                name.startswith(f"base_model.{base_name}") for base_name, _ in sft_model.base_model.named_parameters()
            )
            if is_base_param:
                base_total += param.numel()
                if param.requires_grad:
                    base_trainable += param.numel()
            else:
                probe_total += param.numel()
                if param.requires_grad:
                    probe_trainable += param.numel()
        else:
            # No base model, all parameters are probe parameters
            probe_total += param.numel()
            if param.requires_grad:
                probe_trainable += param.numel()

    def format_params(count: int, total: int) -> str:
        if count >= 1e9:
            return f"{count / 1e9:.2f}B trainable / {total / 1e9:.2f}B total"
        elif count >= 1e6:
            return f"{count / 1e6:.2f}M trainable / {total / 1e6:.2f}M total"
        elif count >= 1e3:
            return f"{count / 1e3:.2f}K trainable / {total / 1e3:.2f}K total"
        else:
            return f"{count} trainable / {total} total"

    logger.info(f"Online training model → {format_params(trainable_params, total_params)}")
    logger.info(f"  Probe parameters: {format_params(probe_trainable, probe_total)}")
    if base_total > 0:
        logger.info(f"  Base model parameters: {format_params(base_trainable, base_total)}")
    sft_model.train()

    # Create optimizer
    optim = get_optimizer(sft_model.parameters(), eval_cfg.training_params)

    # Store probe model in exp_logger for later access (e.g., printing learned weights)
    exp_logger.probe_model = sft_model

    # Create trainer
    trainer = FineTuneTrainer(
        model=sft_model,
        optimizer=optim,
        train_loader=train_dl_raw,
        val_loader=val_dl_raw,
        device=device,
        cfg=eval_cfg,
        exp_logger=exp_logger,
        num_labels=num_labels,
        multi_label=multi_label,
        dataset_metrics=dataset_metrics,
        warmup_epochs=5,  # 5 epochs warmup
        scheduler_type=eval_cfg.training_params.scheduler_type,  # Use scheduler type
        # from config
    )

    # Train
    train_metrics, val_metrics = trainer.train(num_epochs=eval_cfg.training_params.train_epochs)

    # Evaluate on test set
    sft_model.eval()  # Use the trained fine-tuned model, not the base model
    test_metrics = {}

    # Get metric class based on task type and dataset metrics
    if dataset_metrics is not None and len(dataset_metrics) > 0:
        metric_names = dataset_metrics
    else:
        raise ValueError("Expected metrics to be specified in the evaluation configuration. ")

    metrics = [get_metric_class(m, num_labels) for m in metric_names]

    with torch.no_grad():
        for batch in test_dl_raw:
            wav = batch["raw_wav"].to(device)
            mask = batch.get("padding_mask")
            if mask is not None:
                mask = mask.to(device)
            y = batch["label"].to(device)

            logits = sft_model(wav, padding_mask=mask)  # Use sft_model instead of base_model
            for met in metrics:
                met.update(logits, y)

    test_metrics = {name: met.get_primary_metric() for name, met in zip(metric_names, metrics, strict=False)}

    return train_metrics, val_metrics, test_metrics
