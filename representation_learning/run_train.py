"""
Entry‑point script for training experiments.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from representation_learning.configs import (  # type: ignore
    RunConfig,
    load_config,
)
from representation_learning.data.dataset import (  # returns (train_dl, val_dl)
    build_dataloaders,
)
from representation_learning.models.get_model import get_model
from representation_learning.training.distributed import init_distributed
from representation_learning.training.optimisers import get_optimizer
from representation_learning.training.train import Trainer
from representation_learning.utils import ExperimentLogger

logger = logging.getLogger("run_train")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
)


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
    config: RunConfig = load_config(config_path, config_type="run")
    logger.info(f"Loaded config from {config_path}")

    # Initialize distributed training if needed
    local_rank, world_size, is_distributed = init_distributed(
        port=config.distributed_port,
        backend=config.distributed_backend,
    )
    device = torch.device(f"cuda:{local_rank}" if is_distributed else config.device)

    torch.manual_seed(config.seed)

    # 2. Build the dataloaders.
    train_dl, val_dl, _ = build_dataloaders(cfg, device=device)
    logger.info(
        "Dataset ready: %d training batches / %d validation batches",
        len(train_dl),
        len(val_dl),
    )

    # Retrieve the number of labels from the training dataset
    # (Even if not needed for model type.)
    num_labels = len(train_dl.dataset.label2idx)
    logger.info("Number of labels: %d", num_labels)

    # Build the model
    model = get_model(config.model_spec, num_classes=num_labels).to(device)
    logger.info("Model → %s parameters", sum(p.numel() for p in model.parameters()))

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save the config
    with open(output_dir / "config.yml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f)

    # Create experiment logger
    exp_logger = ExperimentLogger.from_config(config)

    # Create optimizer
    optim = get_optimizer(model.parameters(), config.training_params)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optim,
        train_dl=train_dl,
        eval_dl=val_dl,
        model_dir=output_dir / "checkpoints",
        local_rank=local_rank,
        world_size=world_size,
        is_distributed=is_distributed,
        criterion=config.loss_function,
        lr=config.training_params.lr,
        weight_decay=config.training_params.weight_decay,
        max_epochs=config.training_params.train_epochs,
        amp=config.training_params.amp,
        amp_dtype=config.training_params.amp_dtype,
        scheduler_config=config.scheduler.model_dump(mode="json"),
        is_clip_mode=(config.label_type == "text"),
        checkpoint_freq=getattr(config, "checkpoint_freq", 1),
        exp_logger=exp_logger,
        batch_size=config.training_params.batch_size,
        device=device,
        resume_from_checkpoint=getattr(config, "resume_from_checkpoint", None),
        run_config=config,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
