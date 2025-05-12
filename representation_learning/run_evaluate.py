"""
Entry-point script for linear probing/fine-tuning experiments.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from representation_learning.configs import (  # type: ignore
    EvaluateConfig,
    RunConfig,
    load_config,
)
from representation_learning.data.dataset import (  # returns (train_dl, val_dl)
    build_dataloaders,
)
from representation_learning.metrics.metric_factory import get_metric_class
from representation_learning.models.get_model import get_model
from representation_learning.models.linear_probe import LinearProbe
from representation_learning.training.optimisers import get_optimizer
from representation_learning.training.train import FineTuneTrainer
from representation_learning.utils import ExperimentLogger

logger = logging.getLogger("run_finetune")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
)


@dataclass
class ExperimentResult:
    dataset_name: str
    experiment_name: str
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear probe/fine-tune an audio representation model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the evaluation config YAML (see configs/evaluation_configs/*)",
    )
    return parser.parse_args()


def run_experiment(
    eval_cfg: EvaluateConfig,
    dataset_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    device: torch.device,
    save_dir: Path,
) -> ExperimentResult:
    """
    Run a single experiment on a single dataset.

    Args:
        eval_cfg: Evaluation configuration containing training parameters
        dataset_config: Configuration for the dataset
        experiment_config: Configuration for the experiment
        device: Device to run on
        save_dir: Directory to save results

    Returns:
        Experiment results

    Raises
    ------
    FileNotFoundError
        If a required checkpoint cannot be found when `pretrained` is False and
        no valid `checkpoint_path` is supplied in the experiment configuration.
    """
    dataset_name = dataset_config.dataset_name
    experiment_name = experiment_config.run_name

    logger.info(
        "Running experiment '%s' on dataset '%s'", experiment_name, dataset_name
    )

    # Load run config for the experiment
    original_run_cfg: RunConfig = load_config(experiment_config.run_config)
    original_run_cfg.model_spec.audio_config.window_selection = "center"
    original_run_cfg.training_params = eval_cfg.training_params

    # ------------------------------------------------------------------ #
    #  Disable training-time augmentations (e.g. mixup, noise) for evaluation
    # ------------------------------------------------------------------ #
    if hasattr(original_run_cfg, "augmentations"):
        original_run_cfg.augmentations = []
    else:
        original_run_cfg.augmentations = []

    # add sample rate to dataset config
    dataset_config.sample_rate = original_run_cfg.model_spec.audio_config.sample_rate

    # Build the dataloaders
    logger.info("Building dataloaders...")
    train_dl, val_dl, test_dl = build_dataloaders(
        original_run_cfg,
        dataset_config,
        device=device,
    )
    logger.info(
        "Dataset ready: %d training batches / %d validation batches / %d test batches",
        len(train_dl),
        len(val_dl),
        len(test_dl),
    )

    # Log memory usage before loading first batch
    if torch.cuda.is_available():
        logger.info(
            "GPU Memory before loading: %d MB",
            torch.cuda.memory_allocated() / 1024 / 1024,
        )
    else:
        import psutil

        process = psutil.Process()
        logger.info(
            "RAM Usage before loading: %d MB", process.memory_info().rss / 1024 / 1024
        )

    # Get number of classes
    num_labels = len(train_dl.dataset.label2idx)
    logger.info("Number of labels: %d", num_labels)

    base_model = get_model(original_run_cfg.model_spec, num_classes=num_labels).to(
        device
    )

    # If pretrained=True, we don't need to load a checkpoint
    if not experiment_config.pretrained:
        # Determine the checkpoint path: prefer the one specified in the experiment
        # config, otherwise fall back to the default location.
        if getattr(experiment_config, "checkpoint_path", None):
            ckpt_path = Path(experiment_config.checkpoint_path).expanduser()
        else:
            ckpt_path = Path("checkpoints") / "best.pt"

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}. "
                "Specify `checkpoint_path` in the experiment config if the "
                "checkpoint lives elsewhere ."
            )

        # Load checkpoint
        base_model.load_state_dict(
            torch.load(ckpt_path, map_location=device)["model_state_dict"]
        )
        logger.info("Loaded model checkpoint from %s", ckpt_path)

    base_model.eval()
    logger.info(
        "Model → %s parameters", sum(p.numel() for p in base_model.parameters())
    )

    # 5. Get layer names for embedding extraction
    if experiment_config.layers == "last_layer":
        layer_names = [
            name
            for name, module in base_model.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]
        layer_names = [layer_names[-1]]
    else:
        layer_names = experiment_config.layers.split(",")
    logger.info("Layers: %s", layer_names)

    # Create linear probe
    linear_probe = LinearProbe(base_model, layer_names, num_labels, device=device)
    logger.info(
        "Linear probe → %s parameters",
        sum(p.numel() for p in linear_probe.parameters()),
    )

    # ------------------------------------------------------------------ #
    #  Freeze backbone if requested and build optimizer on trainable params
    # ------------------------------------------------------------------ #

    if eval_cfg.frozen:
        logger.info("Freezing backbone parameters (eval_cfg.frozen=True)")
        for p in base_model.parameters():
            p.requires_grad = False

    trainable_params = filter(lambda p: p.requires_grad, linear_probe.parameters())
    optim = get_optimizer(trainable_params, eval_cfg.training_params)

    # Create experiment-specific logger
    exp_logger = ExperimentLogger.from_config(experiment_config)
    exp_logger.log_dir = save_dir / dataset_name / experiment_name
    exp_logger.log_dir.mkdir(parents=True, exist_ok=True)

    trainer = FineTuneTrainer(
        model=linear_probe,
        optimizer=optim,
        train_loader=train_dl,
        val_loader=val_dl,
        device=device,
        cfg=eval_cfg,
        exp_logger=exp_logger,
        multi_label=dataset_config.multi_label,
    )

    # Train the linear probe
    train_metrics, val_metrics = trainer.train(
        num_epochs=eval_cfg.training_params.train_epochs
    )

    # Compute test metrics
    linear_probe.eval()

    # Get metrics from dataset config
    metric_names = dataset_config.metrics
    metrics = [get_metric_class(name.strip(), num_labels) for name in metric_names]

    with torch.no_grad():
        for batch in test_dl:
            x = batch["raw_wav"].to(device)
            mask = batch.get("padding_mask")
            if mask is not None:
                mask = mask.to(device)
            y = batch["label"].to(device)

            # Forward pass
            logits = (
                linear_probe(x, padding_mask=mask)
                if mask is not None
                else linear_probe(x)
            )

            # Update all metrics
            for metric in metrics:
                metric.update(logits, y)

    # Get final test metrics
    test_metrics = {}
    for metric_name, metric in zip(metric_names, metrics, strict=False):
        metric_name = metric_name.strip()
        test_metrics[metric_name] = metric.get_primary_metric()

    # ------------------------------------------------------------------ #
    #  Persist results to the ExperimentLogger backend
    # ------------------------------------------------------------------ #

    exp_logger.log_metrics(train_metrics, step=0, split="train_final")
    exp_logger.log_metrics(val_metrics, step=0, split="val_final")
    exp_logger.log_metrics(test_metrics, step=0, split="test")

    # Flush & close backend; no further logging in this run.
    exp_logger.finalize()

    return ExperimentResult(
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def main() -> None:
    args = _parse_args()

    # 1. Load evaluation config
    eval_cfg: EvaluateConfig = load_config(args.config, config_type="evaluate")
    logger.info("Loaded evaluation config from %s", args.config)

    # 2. Load dataset config
    dataset_cfg = load_config(eval_cfg.dataset_config, config_type="benchmark")
    logger.info("Loaded benchmark dataset config from %s", eval_cfg.dataset_config)

    # 3. Create save directory
    save_dir = Path(eval_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4. Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)  # Fixed seed for reproducibility

    # 5. Run experiments for each dataset and experiment combination
    for dataset in dataset_cfg.datasets:
        results = []
        for experiment in eval_cfg.experiments:
            result = run_experiment(eval_cfg, dataset, experiment, device, save_dir)
            results.append(result)

            # Log results
            logger.info(
                "Results for dataset '%s', experiment '%s':\n"
                "  Train: loss=%.4f, acc=%.4f\n"
                "  Val:   loss=%.4f, acc=%.4f\n"
                "  Test:  %s",
                result.dataset_name,
                result.experiment_name,
                result.train_metrics["loss"],
                result.train_metrics["acc"],
                result.val_metrics["loss"],
                result.val_metrics["acc"],
                result.test_metrics,
            )

    # 6. Save summary of all results
    summary_path = save_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Experiment Summary\n")
        f.write("================\n\n")
        for result in results:
            f.write(f"Dataset: {result.dataset_name}\n")
            f.write(f"Experiment: {result.experiment_name}\n")
            f.write(f"Train metrics: {result.train_metrics}\n")
            f.write(f"Validation metrics: {result.val_metrics}\n")
            f.write(f"Test metrics: {result.test_metrics}\n")
            f.write("-" * 50 + "\n")

    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
