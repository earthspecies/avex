"""
Entry‑point script for training experiments.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from representation_learning.configs import RunConfig, load_config  # type: ignore
from representation_learning.data.dataset import build_dataloaders
from representation_learning.models.get_model import get_model
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
        type=Path,
        required=True,
        help="Path to the run‑config YAML (see configs/*)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # 1. Load & validate config
    cfg: RunConfig = load_config(args.config)
    logger.info("Loaded run config from %s", args.config)

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    # 2. Build the dataloaders.
    train_dl, val_dl = build_dataloaders(cfg, device=device)
    logger.info(
        "Dataset ready: %d training batches / %d validation batches",
        len(train_dl),
        len(val_dl),
    )

    # 3. Retrieve the number of labels from the training dataset (Even if not needed for
    #    model type.)
    num_labels = train_dl.dataset.metadata["num_classes"]
    logger.info("Number of labels: %d", num_labels)

    # 4. Build the model
    model = get_model(cfg.model_spec, num_classes=num_labels).to(device)
    logger.info("Model → %s parameters", sum(p.numel() for p in model.parameters()))

    optim = get_optimizer(model.parameters(), cfg.training_params)
    exp_logger = ExperimentLogger.from_config(cfg)

    trainer = Trainer(
        model=model,
        optimizer=optim,
        train_loader=train_dl,
        val_loader=val_dl,
        device=device,
        cfg=cfg,
        exp_logger=exp_logger,
    )

    trainer.train()


if __name__ == "__main__":
    main()
