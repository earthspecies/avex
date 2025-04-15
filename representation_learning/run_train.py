"""representation_learning/run_train.py
Entry‑point script for training experiments.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from representation_learning.training.optimisers import get_optimizer
import torch

# --------------------------------------------------------------------------- #
#  Internal imports (make sure representation_learning is on PYTHONPATH)
# --------------------------------------------------------------------------- #
from representation_learning.configs import load_config, RunConfig  # type: ignore
from representation_learning.models.base_model import get_model                # factory in models/__init__.py
from representation_learning.data.data_utils import build_dataloaders  # returns (train_dl, val_dl)
from representation_learning.training.train import Trainer         # high‑level training loop

logger = logging.getLogger("run_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an audio representation model")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the run‑config YAML (see configs/*)"
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = _parse_args()

    # 1. Load & validate config ------------------------------------------------
    cfg: RunConfig = load_config(args.config)
    logger.info("Loaded run config from %s", args.config)

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)


    model = get_model(cfg.model_config, cfg).to(device)
    logger.info("Model → %s parameters", sum(p.numel() for p in model.parameters()))

    train_dl, val_dl = build_dataloaders(cfg, device=device)
    logger.info("Dataset ready: %d training batches / %d validation batches", len(train_dl), len(val_dl))

    optim = get_optimizer(cfg.training_params)

    trainer = Trainer(
        model=model,
        optimizer=optim,
        train_loader=train_dl,
        val_loader=val_dl,
        device=device,
        cfg=cfg,
    )

    trainer.train(num_epochs=cfg.training_params.train_epochs)


if __name__ == "__main__":
    main()
