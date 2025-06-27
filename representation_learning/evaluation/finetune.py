import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from representation_learning.configs import EvaluateConfig
from representation_learning.metrics.metric_factory import get_metric_class
from representation_learning.metrics.strong_detection.detection_metric_helpers import (
    _frames_to_events,
)
from representation_learning.models.framewise_linear_probe import FramewiseLinearProbe
from representation_learning.models.linear_probe import LinearProbe
from representation_learning.training.optimisers import get_optimizer
from representation_learning.training.train import FineTuneTrainer
from representation_learning.utils import ExperimentLogger

logger = logging.getLogger("run_finetune")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


# -------------------------------------------------------------------- #
#  Linear-probe helper
# -------------------------------------------------------------------- #
def train_and_eval_linear_probe(
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    test_embed_ds: torch.utils.data.Dataset,
    base_model: Optional[torch.nn.Module],
    num_labels: int,
    layer_names: List[str],
    eval_cfg: EvaluateConfig,
    device: torch.device,
    exp_logger: ExperimentLogger,
    multi_label: bool,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Train a linear probe and evaluate it on *cached* test embeddings.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
        • **train_metrics** – aggregated over training split
        • **val_metrics**   – aggregated over validation split
        • **probe_test_metrics** – metrics on cached-embedding test split
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

    if base_model is not None:
        for p in base_model.parameters():
            p.requires_grad = False

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
        multi_label=multi_label,
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

    # Metric selection (fallback to accuracy)
    metric_names = getattr(test_embed_ds, "metadata", {}).get("metrics", ["accuracy"])
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
#  Framewise linear probe helper for strong detection
# -------------------------------------------------------------------- #
def train_and_eval_framewise_probe(
    train_dl_raw: torch.utils.data.DataLoader,
    val_dl_raw: torch.utils.data.DataLoader,
    test_dl_raw: torch.utils.data.DataLoader,
    base_model: torch.nn.Module,
    num_labels: int,
    layer_names: List[str],
    eval_cfg: EvaluateConfig,
    device: torch.device,
    exp_logger: ExperimentLogger,
    metric_names: List[str],
    fps: float = 50.0,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Train a framewise linear probe for strong detection evaluation.

    Parameters
    ----------
    train_dl_raw : torch.utils.data.DataLoader
        Training dataloader with raw waveforms and frame-level targets
    val_dl_raw : torch.utils.data.DataLoader
        Validation dataloader with raw waveforms and frame-level targets
    test_dl_raw : torch.utils.data.DataLoader
        Test dataloader with raw waveforms and frame-level targets
    base_model : torch.nn.Module
        Frozen backbone network
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
    metric_names : List[str]
        Names of metrics to compute

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
        • **train_metrics** – aggregated over training split
        • **val_metrics**   – aggregated over validation split
        • **probe_test_metrics** – metrics on test split
    """

    # Create framewise linear probe
    probe = FramewiseLinearProbe(
        base_model=base_model,
        layers=layer_names,
        num_classes=num_labels,
        device=device,
        feature_mode=False,  # We'll extract embeddings inside the probe
    )

    logger.info(
        "Framewise probe → %d parameters", sum(p.numel() for p in probe.parameters())
    )

    # Freeze the base model (probe backbone)
    for p in base_model.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # 1. Training & Validation via FineTuneTrainer
    # ------------------------------------------------------------------
    optim = get_optimizer(probe.classifier.parameters(), eval_cfg.training_params)

    trainer = FineTuneTrainer(
        model=probe,
        optimizer=optim,
        train_loader=train_dl_raw,
        val_loader=val_dl_raw,
        device=device,
        cfg=eval_cfg,
        exp_logger=exp_logger,
        multi_label=True,  # Frame-wise strong detection is multi-label
    )

    train_metrics, val_metrics = trainer.train(
        num_epochs=eval_cfg.training_params.train_epochs
    )

    # ------------------------------------------------------------------
    # 2. Test-set evaluation – strong detection metrics
    # ------------------------------------------------------------------
    probe.eval()
    metrics = [get_metric_class(m, num_labels) for m in metric_names]

    # Propagate fps to metrics that rely on it (e.g. StrongDetectionF1Tensor)
    for name, met in zip(metric_names, metrics, strict=False):
        if name == "f1_strong" and hasattr(met, "fps"):
            met.fps = fps

    with torch.no_grad():
        for batch in test_dl_raw:
            wav = batch["raw_wav"].to(device)
            targets = batch["frame_targets"].to(device)  # (B, T, C)

            padding_mask = batch.get("padding_mask")
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            # Forward pass → (B, T, C)
            logits = probe(wav, padding_mask)

            # ----------------------------------------------------------
            # Convert reference frames → event lists per sample × class
            # ----------------------------------------------------------
            targets_events: list[list[np.ndarray]] = []  # type: ignore[name-defined]
            for b in range(targets.shape[0]):
                valid_mask = (
                    (~padding_mask[b]).cpu().numpy()
                    if padding_mask is not None
                    else None
                )

                batch_events: list[np.ndarray] = []  # type: ignore[name-defined]
                for c in range(targets.shape[2]):
                    frames_np = targets[b, :, c].cpu().numpy()
                    if valid_mask is not None:
                        frames_np = frames_np[valid_mask]
                    batch_events.append(_frames_to_events(frames_np, fps))
                targets_events.append(batch_events)

            # Update strong-detection metric(s)
            for name, met in zip(metric_names, metrics, strict=False):
                if name == "f1_strong":
                    met.update(logits, targets_events, padding_mask)
                else:
                    # Fallback: average over time → clip-level
                    met.update(logits.mean(dim=1), targets.mean(dim=1))

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

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
        • **train_metrics** – aggregated over training split
        • **val_metrics**   – aggregated over validation split
        • **probe_test_metrics** – metrics on test split
    """
    # Enable training mode
    base_model.train()
    for p in base_model.parameters():
        p.requires_grad = True

    # Create optimizer
    optim = get_optimizer(base_model.parameters(), eval_cfg.training_params)

    # Create trainer
    trainer = FineTuneTrainer(
        model=base_model,
        optimizer=optim,
        train_loader=train_dl_raw,
        val_loader=val_dl_raw,
        device=device,
        cfg=eval_cfg,
        exp_logger=exp_logger,
        multi_label=multi_label,
    )

    # Train
    train_metrics, val_metrics = trainer.train(
        num_epochs=eval_cfg.training_params.train_epochs
    )

    # Evaluate on test set
    base_model.eval()
    test_metrics = {}

    # Get metric class based on task type
    metric_names = ["accuracy"] if not multi_label else ["f1"]
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
