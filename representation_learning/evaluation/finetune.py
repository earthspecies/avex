import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from representation_learning.configs import EvaluateConfig
from representation_learning.metrics.metric_factory import get_metric_class
from representation_learning.models.linear_probe import LinearProbe
from representation_learning.training.optimisers import get_optimizer
from representation_learning.utils import ExperimentLogger

logger = logging.getLogger("run_finetune")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


# -------------------------------------------------------------------- #
#  FineTuneTrainer (moved from dep_train.py)
# -------------------------------------------------------------------- #
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
        num_labels: int,
        multi_label: bool = False,
        dataset_metrics: Optional[List[str]] = None,
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

        # Determine primary metric name based on task type and dataset configuration
        if self.dataset_metrics and len(self.dataset_metrics) > 0:
            self.primary_metric_name = self.dataset_metrics[0].lower()
        else:
            # For detection tasks (multi_label=True), require explicit metrics
            if self.multi_label:
                raise ValueError("No dataset metrics provided for multi-label task. ")
            else:
                # For single-label classification, still allow accuracy as default
                self.primary_metric_name = "acc"

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

    def train(self, num_epochs: int) -> tuple[dict[str, float], dict[str, float]]:
        """
        Run the full fine-tuning loop and return best train/val metrics.

        Returns
        -------
        tuple[dict[str, float], dict[str, float]]
            A tuple of (best_train_metrics, best_val_metrics) collected across epochs.
        """
        for epoch in range(1, num_epochs + 1):
            train_loss, train_metric = self._run_epoch(train=True, epoch=epoch)
            val_loss, val_metric = self._run_epoch(train=False, epoch=epoch)

            logger.info(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f}  "
                f"train_{self.primary_metric_name}={train_metric:.4f} | "
                f"val_loss={val_loss:.4f}  "
                f"val_{self.primary_metric_name}={val_metric:.4f}"
            )

            # Log epoch-level metrics
            self.log.log_metrics(
                {"loss": train_loss, self.primary_metric_name: train_metric},
                step=epoch,
                split="train",
            )
            self.log.log_metrics(
                {"loss": val_loss, self.primary_metric_name: val_metric},
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
            metric_calculator = get_metric_class(
                self.primary_metric_name, self.num_labels
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize metric '{self.primary_metric_name}' "
                f"for {self.num_labels} classes. "
                f"Error: {e}. This indicates a configuration issue - ensure "
                f"the metric name is valid "
                f"and properly supported by the metric factory."
            ) from e

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
                self.optimizer.step()

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
            f"Restored best model from {ckpt_path} "
            f"(val_{self.primary_metric_name}: {self.best_val_metric:.4f})"
        )


# -------------------------------------------------------------------- #
#  Linear-probe helper
# -------------------------------------------------------------------- #
def train_and_eval_linear_probe(
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    test_embed_ds: torch.utils.data.Dataset,
    num_labels: int,
    layer_names: List[str],
    eval_cfg: EvaluateConfig,
    device: torch.device,
    exp_logger: ExperimentLogger,
    multi_label: bool,
    dataset_metrics: Optional[List[str]] = None,
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

    # Get input dimension from the first batch of training data
    first_batch = next(iter(torch.utils.data.DataLoader(train_ds, batch_size=1)))
    input_dim = first_batch["embed"].shape[1]

    probe = LinearProbe(
        base_model=None,
        layers=layer_names,
        num_classes=num_labels,
        device=device,
        feature_mode=True,
        input_dim=input_dim,
    )
    logger.info(
        "Linear probe → %d parameters", sum(p.numel() for p in probe.parameters())
    )
    probe.train()

    optim = get_optimizer(probe.parameters(), eval_cfg.training_params)

    trainer = FineTuneTrainer(
        model=probe,
        optimizer=optim,
        train_loader=torch.utils.data.DataLoader(
            train_ds,
            batch_size=eval_cfg.training_params.batch_size,
            shuffle=True,
        ),
        val_loader=torch.utils.data.DataLoader(
            val_ds,
            batch_size=eval_cfg.training_params.batch_size,
            shuffle=False,
        ),
        device=device,
        cfg=eval_cfg,
        exp_logger=exp_logger,
        num_labels=num_labels,
        multi_label=multi_label,
        dataset_metrics=dataset_metrics,
    )

    train_metrics, val_metrics = trainer.train(
        num_epochs=eval_cfg.training_params.train_epochs
    )

    # ---------- probe evaluation on cached test embeddings ----------
    test_loader = torch.utils.data.DataLoader(
        test_embed_ds,
        batch_size=eval_cfg.training_params.batch_size,
        shuffle=False,
    )

    # Metric selection - require explicit metrics, no fallbacks
    if dataset_metrics is not None and len(dataset_metrics) > 0:
        metric_names = dataset_metrics
    else:
        raise ValueError(
            "Expected metrics to be specified in the evaluation configuration. "
        )

    metrics = [get_metric_class(m, num_labels) for m in metric_names]

    probe.eval()  # feature_mode stays True (inputs are embeddings)
    with torch.no_grad():
        for batch in test_loader:
            z = batch["embed"].to(device)
            y = batch["label"].to(device)

            logits = probe(z)
            for met in metrics:
                met.update(logits, y)

    probe_test_metrics = {
        name: met.get_primary_metric()
        for name, met in zip(metric_names, metrics, strict=False)
    }
    return train_metrics, val_metrics, probe_test_metrics


# -------------------------------------------------------------------- #
#  Full fine-tune helper
# -------------------------------------------------------------------- #
def train_and_eval_full_fine_tune(
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
    # Enable training mode
    base_model.train()
    for p in base_model.parameters():
        p.requires_grad = True

    sft_model = LinearProbe(
        base_model=base_model,
        layers=layer_names,
        num_classes=num_labels,
        device=device,
        feature_mode=False,
        input_dim=None,
    )
    logger.info(
        "Fully fine-tuned model → %d parameters",
        sum(p.numel() for p in sft_model.parameters()),
    )
    sft_model.train()

    # Create optimizer
    optim = get_optimizer(sft_model.parameters(), eval_cfg.training_params)

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
    )

    # Train
    train_metrics, val_metrics = trainer.train(
        num_epochs=eval_cfg.training_params.train_epochs
    )

    # Evaluate on test set
    base_model.eval()
    test_metrics = {}

    # Get metric class based on task type and dataset metrics
    if dataset_metrics is not None and len(dataset_metrics) > 0:
        metric_names = dataset_metrics
    else:
        raise ValueError(
            "Expected metrics to be specified in the evaluation configuration. "
        )

    metrics = [get_metric_class(m, num_labels) for m in metric_names]

    with torch.no_grad():
        for batch in test_dl_raw:
            wav = batch["raw_wav"].to(device)
            mask = batch.get("padding_mask")
            if mask is not None:
                mask = mask.to(device)
            y = batch["label"].to(device)

            logits = base_model(wav, padding_mask=mask)
            for met in metrics:
                met.update(logits, y)

    test_metrics = {
        name: met.get_primary_metric()
        for name, met in zip(metric_names, metrics, strict=False)
    }

    return train_metrics, val_metrics, test_metrics
