import logging
from typing import Dict, List, Optional, Tuple

import torch

from representation_learning.configs import EvaluateConfig
from representation_learning.metrics.metric_factory import get_metric_class
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
            "This indicates a configuration error - metrics should be explicitly "
            "defined for each evaluation set to avoid silent bugs."
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

    # Get metric class based on task type and dataset metrics - require explicit
    # metrics, no fallbacks
    if dataset_metrics is not None and len(dataset_metrics) > 0:
        metric_names = dataset_metrics
    else:
        raise ValueError(
            "Expected metrics to be specified in the evaluation configuration. "
            "This indicates a configuration error - metrics should be explicitly "
            "defined for each evaluation set to avoid silent bugs."
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
