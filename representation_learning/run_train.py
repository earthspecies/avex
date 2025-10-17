"""Training script for representation learning models."""

import json
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import torch
import yaml
from esp_data.io.paths import anypath  # type: ignore

from representation_learning.configs import RunConfig
from representation_learning.data.dataset import build_dataloaders
from representation_learning.models.get_model import get_model
from representation_learning.training.distributed import (
    get_local_device_index,
    init_distributed,
)
from representation_learning.training.optimisers import (
    get_optimizer,
)
from representation_learning.training.trainer_factory import (
    TrainerFactory,
)
from representation_learning.training.training_utils import (
    build_scheduler,
)
from representation_learning.utils import ExperimentLogger

# Configure multiprocessing
mp.set_start_method("spawn", force=True)


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

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

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

    # --------------------------------------------------------------
    # Optional 1st-stage backbone freeze for two-stage fine-tuning
    # --------------------------------------------------------------
    freeze_epochs = getattr(config.training_params, "freeze_backbone_epochs", 0)
    if freeze_epochs > 0 and hasattr(model, "backbone"):
        logger.info("Freezing backbone for the first %d epochs", freeze_epochs)
        for p in model.backbone.parameters():  # type: ignore[attr-defined]
            p.requires_grad = False

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

    # Save label_map for reference
    label_map = train_dl.dataset.metadata.get("label_map", {})
    if label_map:
        with (output_dir / "label_map.json").open("w") as f:
            json.dump(label_map, f, indent=2)
        logger.info(
            f"Saved label_map with {len(label_map)} classes to "
            f"{output_dir / 'label_map.json'}"
        )
    else:
        logger.warning("No label_map found in dataset metadata")

    # Create experiment logger
    exp_logger = ExperimentLogger.from_config(config)

    # Create optimizer
    optim = get_optimizer(model.parameters(), config.training_params)

    # Create scheduler
    total_steps = len(train_dl) * config.training_params.train_epochs
    scheduler = build_scheduler(optim, config, total_steps)

    # Create scaler for mixed precision training
    scaler = None
    if config.training_params.amp:
        from torch.cuda.amp import GradScaler

        scaler = GradScaler()

    # Keep the original config.output_dir which was already set correctly above

    # Create trainer using the factory
    trainer = TrainerFactory.create_trainer(
        model=model,
        optimizer=optim,
        scheduler=scheduler,
        scaler=scaler,
        train_dataloader=train_dl,
        eval_dataloader=val_dl,
        config=config,
        local_rank=local_device_index,
        world_size=world_size,
        is_distributed=is_distributed,
        device=device,
        exp_logger=exp_logger,
        num_classes=num_labels,
        resume_from_checkpoint=getattr(config, "resume_from_checkpoint", None),
    )

    # Train
    trainer.train()
