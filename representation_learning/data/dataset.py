from __future__ import annotations

import multiprocessing
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from esp_data_temp.dataset import get_dataset_dummy
from representation_learning.configs import RunConfig, load_config
from representation_learning.data.audio_utils import (
    pad_or_window,  # type: ignore
)
from representation_learning.data.augmentations import (
    AugmentationProcessor,
    make_item_postprocessor,
)


# --------------------------------------------------------------------------- #
#  Collater
# --------------------------------------------------------------------------- #
class Collater:
    """
    Combines samples into a batch, ensuring every audio clip has the same
    length (`audio_max_length`) by truncating or zero‑padding as needed.
    """

    def __init__(
        self,
        audio_max_length_seconds: int,
        sr: int,
        window_selection: str = "random",
        keep_text: bool = False,
        preprocessor: Optional[str] = None,
        device: str = "cpu",
        batch_aug_processor: Optional[AugmentationProcessor] = None,
    ) -> None:
        self.audio_max_length_seconds = audio_max_length_seconds
        self.window_selection = window_selection
        self.keep_text = keep_text
        self.preprocessor = preprocessor
        self.sr = sr
        self.device = device
        self.batch_aug_processor = batch_aug_processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # First prepare data with uniform lengths
        audios, masks, labels, text_labels = [], [], [], []

        for item in batch:
            wav = torch.as_tensor(item["raw_wav"])  # (T,)
            wav, pad_mask = pad_or_window(
                wav, self.audio_max_length_seconds * self.sr, self.window_selection
            )
            audios.append(wav)
            masks.append(pad_mask)
            labels.append(item["label"])
            if self.keep_text:
                text_labels.append(item["text_label"])

        # Apply batch-level mixup AFTER all audios are same length
        if self.batch_aug_processor is not None and audios:
            # Create temporary batch of uniform-length samples
            temp_batch = [
                {
                    "raw_wav": wav,
                    "label": lbl,
                    "text_label": txt if self.keep_text else None,
                }
                for wav, lbl, txt in zip(
                    audios,
                    labels,
                    text_labels if self.keep_text else [None] * len(audios),
                    strict=False,
                )
            ]
            # Now mixup can safely operate on same-sized tensors
            mixed_batch = self.batch_aug_processor.apply_batch_augmentations(temp_batch)
            # Extract back to separate lists
            audios = [item["raw_wav"] for item in mixed_batch]
            labels = [item["label"] for item in mixed_batch]
            if self.keep_text:
                text_labels = [item["text_label"] for item in mixed_batch]

        # ------------------------------------
        # Stack into tensors (audio + mask)
        # ------------------------------------

        audio_tensor = torch.stack(audios)  # [B, T] float32
        mask_tensor = torch.stack(masks)  # [B, T] bool

        # Handle different label formats (int → long, multi-hot → float32)
        if all(isinstance(lbl, (int, np.integer)) for lbl in labels):
            label_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            # Convert each label (multi-hot) to tensor and stack along batch dim
            label_tensors = [
                torch.as_tensor(lbl, dtype=torch.float32)
                if not isinstance(lbl, torch.Tensor)
                else lbl.float()
                for lbl in labels
            ]
            label_tensor = torch.stack(label_tensors)

        return {
            "raw_wav": audio_tensor,
            "padding_mask": mask_tensor,
            "label": label_tensor,
            "text_label": text_labels,
        }


# Function moved outside of build_dataloaders to make it picklable
def worker_init_fn(worker_id: int) -> None:
    """Initialize worker process with proper logging and seeding.

    This function is called by PyTorch DataLoader workers when they start.
    """
    import logging
    import random

    import numpy as np
    import torch

    from representation_learning.data.augmentations import _cached_audio_info

    # Get global variables from the parent process
    # These will be automatically inherited by worker processes through globals
    # Configure worker-specific logging
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | Worker-%(process)d | %(name)s: %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("worker_init")

    # Access the global worker_init_data dictionary
    global _worker_init_data
    if "_worker_init_data" not in globals():
        logger.warning(
            f"Worker {worker_id}: No _worker_init_data found. Skipping initialization."
        )
        return

    data = _worker_init_data
    seed = data.get("seed", 42)
    aug_processor = data.get("aug_processor")

    # Per-worker seed for deterministic but varied randomization
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    logger.debug(f"Worker {worker_id} initialized with seed {worker_seed}")

    # Prefetch audio info cache for noise augmentations if this is worker 0
    if (
        worker_id == 0
        and aug_processor is not None
        and hasattr(aug_processor, "_noise_pools")
    ):
        logger.info(f"Worker {worker_id}: Prefetching audio info for noise samples")

        # Prefetch a subset of files to avoid taking too long at startup
        for _cfg_id, noise_files in aug_processor._noise_pools.items():
            # Prefetch a maximum of 100 files per noise config to avoid startup delay
            prefetch_count = min(50, len(noise_files))
            for noise_path in noise_files[:prefetch_count]:
                try:
                    _cached_audio_info(str(noise_path))
                except Exception as e:
                    logger.warning(f"Failed to prefetch info for {noise_path}: {e}")
        logger.info(f"Worker {worker_id}: Prefetching complete")


# Global variable to store data for worker initialization
_worker_init_data = {}


def build_dataloaders(
    cfg: RunConfig, device: str = "cpu"
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders from configuration.

    Parameters
    ----------
    cfg : RunConfig
        Run configuration containing dataset and training parameters
    device : str
        Device to use for data loading

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    if device != "cpu":
        multiprocessing.set_start_method("spawn", force=True)

    # Load dataset configuration
    data_config = load_config(cfg.dataset_config, config_type="data")

    postprocessors = []
    aug_processor = None
    if cfg.augmentations:
        aug_device = "cpu"  # dataloader augments on CPU
        aug_processor = AugmentationProcessor(cfg.augmentations, cfg.sr, aug_device)
        postprocessors.append(make_item_postprocessor(aug_processor))

    # Create dataset using the updated get_dataset_dummy
    ds_train = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,
        validation=cfg.debug_mode,
        postprocessors=postprocessors,
    )
    ds_eval = get_dataset_dummy(
        data_config=data_config,
        preprocessor=None,
        validation=True,
        postprocessors=None,
    )

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(ds_train)
        val_sampler = DistributedSampler(ds_eval, shuffle=False)

    # Create collater
    collate_fn_train = Collater(
        audio_max_length_seconds=cfg.model_spec.audio_config.target_length_seconds,
        sr=cfg.model_spec.audio_config.sample_rate,
        window_selection=cfg.model_spec.audio_config.window_selection,
        keep_text=(cfg.label_type == "text"),  # Keep text labels for CLIP training
        device=device,
        batch_aug_processor=aug_processor,
    )

    collate_fn_eval = Collater(
        audio_max_length_seconds=cfg.model_spec.audio_config.target_length_seconds,
        sr=cfg.model_spec.audio_config.sample_rate,
        window_selection=cfg.model_spec.audio_config.window_selection,
        keep_text=(cfg.label_type == "text"),
        device=device,
        batch_aug_processor=None,  # no augmentation during evaluation
    )

    # Store the data needed for worker initialization in a global variable
    global _worker_init_data
    _worker_init_data = {"seed": cfg.seed, "aug_processor": aug_processor}

    # Create dataloaders with optimized settings
    train_dl = DataLoader(
        ds_train,
        batch_size=cfg.training_params.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_train,
        pin_memory=(device != "cpu"),
        worker_init_fn=worker_init_fn,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        drop_last=True,  # Slight optimization for training - drops partial batches
    )

    val_dl = DataLoader(
        ds_eval,
        batch_size=cfg.training_params.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_eval,
        pin_memory=(device != "cpu"),
        worker_init_fn=worker_init_fn,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    return train_dl, val_dl
