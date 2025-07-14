"""
Entry‑point script for training experiments.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Cloud-agnostic path factory (local / gs:// / r2://).
from esp_data.io.paths import anypath  # type: ignore

from representation_learning.configs import (  # type: ignore
    RunConfig,
)
from representation_learning.data.dataset import build_dataloaders
from representation_learning.models.get_model import get_model
from representation_learning.training.distributed import (
    get_local_device_index,
    init_distributed,
)
from representation_learning.training.optimisers import get_optimizer
from representation_learning.training.train import Trainer
from representation_learning.utils import ExperimentLogger

# Configure logging to ensure INFO level logs are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("run_train")


def main(config_path: Path, patches: tuple[str, ...] | None = None) -> None:
    """
    Training entry point.

    Parameters
    ----------
    config_path : Path
        Path to the config file
    patches : tuple[str, ...] | None
        Tuple of config patches in format 'key=value'.
    """

    if patches is None:
        patches = ()
    config = RunConfig.from_sources(yaml_file=str(config_path), cli_args=patches)
    logger.info(f"Loaded config from {config_path}")

    # Initialize distributed training if needed
    local_rank, world_size, is_distributed = init_distributed(
        port=config.distributed_port,
        backend=config.distributed_backend,
    )

    # Get the correct local device index using the improved logic
    if torch.cuda.is_available():
        local_device_index = get_local_device_index()
        torch.cuda.set_device(local_device_index)
        device = torch.device("cuda", local_device_index)
        logger.info(f"Using CUDA device {local_device_index}")
    else:
        local_device_index = 0
        device = torch.device("cpu")
        logger.info("Using CPU device")

    torch.manual_seed(config.seed)

    # 2. Build the dataloaders.
    train_dl, val_dl, _ = build_dataloaders(config, device=device)
    logger.info(
        "Dataset ready: %d training batches / %d validation batches",
        len(train_dl),
        len(val_dl),
    )

    num_labels = len(train_dl.dataset.metadata["label_map"])

    logger.info("Number of labels: %d", num_labels)

    # Enable EAT self-supervised mode when requested
    if config.label_type == "self_supervised":
        # Pydantic models are immutable by default – use copy(update=...)
        config.model_spec = config.model_spec.model_copy(
            update={"pretraining_mode": True}
        )

    # Build the model
    model = get_model(config.model_spec, num_classes=num_labels).to(device)
    logger.info("Model → %s parameters", sum(p.numel() for p in model.parameters()))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base_out = anypath(config.output_dir)
    output_dir = base_out / timestamp  # type: ignore[operator]

    try:
        output_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[arg-type]
    except AttributeError:
        # CloudPath objects may not implement mkdir – will create implicitly
        pass

    config.output_dir = str(output_dir)

    # Save the config (works for cloud & local)
    with (output_dir / "config.yml").open("w") as f:
        yaml.dump(config.model_dump(mode="json"), f)

    # Create experiment logger
    exp_logger = ExperimentLogger.from_config(config)

    # Create optimizer
    optim = get_optimizer(model.parameters(), config.training_params)

    trainer = Trainer(
        model=model,
        optimizer=optim,
        train_dl=train_dl,
        eval_dl=val_dl,
        model_dir=output_dir / "checkpoints",
        local_rank=local_device_index,
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
        is_eat_ssl=(config.label_type == "self_supervised"),
        checkpoint_freq=getattr(config, "checkpoint_freq", 1),
        exp_logger=exp_logger,
        batch_size=config.training_params.batch_size,
        device=device,
        resume_from_checkpoint=getattr(config, "resume_from_checkpoint", None),
        run_config=config,
        log_steps=config.training_params.log_steps,
        gradient_checkpointing=config.training_params.gradient_checkpointing,
    )

    # Train
    trainer.train()
