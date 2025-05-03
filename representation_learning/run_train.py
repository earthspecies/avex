"""
Entry‑point script for training experiments.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch
import os
import yaml

from representation_learning.configs import load_config, RunConfig  # type: ignore
from representation_learning.models.get_model import get_model
from representation_learning.data.dataset import build_dataloaders  # returns (train_dl, val_dl)
from representation_learning.training.train import Trainer
from representation_learning.training.optimisers import get_optimizer
from representation_learning.training.distributed import setup_distributed
from representation_learning.utils import ExperimentLogger

logger = logging.getLogger("run_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an audio representation model")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/run_configs/clip_base.yml",
        help="Path to the config file",
    )
    return parser.parse_args()


def main() -> None:
    """
    Training entry point.
    """
    args = _parse_args()
    config_path = args.config

    # Load config
    config = load_config(config_path, config_type="run")
    logger.info(f"Loaded config from {config_path}")

    # Initialize distributed training if needed
    local_rank, world_size, _ = setup_distributed(
        backend=config.distributed_backend,
        port=config.distributed_port,
    )
    is_distributed = local_rank is not None

    # Set device based on distributed setup
    if is_distributed:
        logger.info("Running in distributed mode with world size %s", world_size)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(config.device)

    torch.manual_seed(config.seed)

    # Create dataloaders
    train_dl, val_dl, augmentation_processor = build_dataloaders(config, device)
    logger.info(
        "Dataset ready: %d training batches / %d validation batches",
        len(train_dl), len(val_dl)
    )

    # Retrieve the number of labels from the training dataset (Even if not needed for model type.)
    num_labels = len(train_dl.dataset.label2idx)
    logger.info("Number of labels: %d", num_labels)

    # Build the model
    model = get_model(config.model_id, config.model_spec, device=device)
    logger.info("Model → %s parameters", sum(p.numel() for p in model.parameters()))

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save the config
    with open(output_dir / "config.yml", "w") as f:
        yaml.dump(config.dict(), f)

    # Create experiment logger
    exp_logger = ExperimentLogger(
        log_dir=output_dir / "logs",
        use_mlflow=config.use_mlflow,
        experiment_name=config.experiment_name,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        eval_dl=val_dl,
        optimizer=config.training_params.optimizer,
        criterion=config.training_params.loss_function,
        batch_size=config.training_params.batch_size,
        lr=config.training_params.learning_rate,
        weight_decay=config.training_params.weight_decay,
        max_epochs=config.training_params.epochs,
        amp=config.training_params.amp,
        amp_dtype=config.training_params.amp_dtype,
        scheduler_config=config.training_params.scheduler,
        model_dir=output_dir / "checkpoints",
        exp_logger=exp_logger,
        is_clip_mode=(config.label_type == "text"),
        checkpoint_freq=config.training_params.checkpoint_freq if hasattr(config.training_params, "checkpoint_freq") else 1,
        augmentation_processor=augmentation_processor,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
