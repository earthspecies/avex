from __future__ import annotations

import multiprocessing
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from esp_data import Dataset, dataset_from_config
from esp_data.transforms import LabelFromFeatureConfig
from torch.utils.data import DataLoader, DistributedSampler

from representation_learning.configs import (
    DatasetConfig,
    RunConfig,
    load_config,
)
from representation_learning.data.audio_utils import (
    pad_or_window,  # type: ignore
)
from representation_learning.data.augmentations import (
    AugmentationProcessor,
    make_item_postprocessor,
)


class AudioDataset(torch.utils.data.Dataset):
    """A wrapper around a Dataset instance for audio data.
    Allows for post-processing of audio samples after retrieval.
    """

    def __init__(self, ds: Dataset, postprocessors: Optional[list[Any]] = None) -> None:
        """Initialize the AudioDataset with a Dataset instance."""
        self.ds = ds
        self.postprocessors = postprocessors or []

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the audio data, text label, label, and path.
        """
        sample = self.ds[idx]

        if self.postprocessors:
            for postprocessor in self.postprocessors:
                sample = postprocessor(sample)

        return sample


# --------------------------------------------------------------------------- #
#  Collater
# --------------------------------------------------------------------------- #
class Collater:
    """
    Combines samples into a batch, ensuring every audio clip has the same
    length (`audio_max_length`) by truncating or zero-padding as needed.
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
        num_labels: int = 0,
    ) -> None:
        self.audio_max_length_seconds = audio_max_length_seconds
        self.window_selection = window_selection
        self.keep_text = keep_text
        self.preprocessor = preprocessor
        self.sr = sr
        self.device = device
        self.batch_aug_processor = batch_aug_processor
        assert num_labels > 1, "num_labels must be greater than 1"
        self.num_labels = num_labels

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
            # Convert integer labels to one-hot vectors for classification
            label_tensor = torch.nn.functional.one_hot(
                torch.tensor(labels, dtype=torch.long), num_classes=self.num_labels
            ).float()
        else:
            # For multi-label case, convert lists of class indices to one-hot vectors
            label_tensors = []
            for lbl in labels:
                # Create zero tensor of size num_labels
                one_hot = torch.zeros(self.num_labels, dtype=torch.float32)
                # Convert list of indices to tensor and set 1s at those indices
                indices = torch.tensor(lbl, dtype=torch.long)
                one_hot[indices] = 1.0
                label_tensors.append(one_hot)
            label_tensor = torch.stack(label_tensors)

        return {
            "raw_wav": audio_tensor,
            "padding_mask": mask_tensor,
            "label": label_tensor,
            "text_label": text_labels,
        }


# --------------------------------------------------------------------------- #
#  Data-loader helpers
# --------------------------------------------------------------------------- #
def worker_init_fn(worker_id: int) -> None:
    """Initialize a DataLoader worker (seeding, logging, audio-info cache)."""
    import logging
    import random

    import numpy as np
    import torch

    from representation_learning.data.augmentations import _cached_audio_info

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | Worker-%(process)d | %(name)s: %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("worker_init")

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

    # Prefetch noise-sample audio headers (worker 0 only, truncated)
    if (
        worker_id == 0
        and aug_processor is not None
        and hasattr(aug_processor, "_noise_pools")
    ):
        logger.info(f"Worker {worker_id}: Prefetching audio info for noise samples")
        for _cfg_id, noise_files in aug_processor._noise_pools.items():
            for noise_path in noise_files[:50]:  # limit to 50 per pool
                try:
                    _cached_audio_info(str(noise_path))
                except Exception as e:
                    logger.warning(f"Failed to prefetch info for {noise_path}: {e}")


# Will be populated before DataLoader construction
_worker_init_data: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
#  Main builder
# --------------------------------------------------------------------------- #
def build_dataloaders(
    cfg: RunConfig,
    data_config: DatasetConfig | None = None,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train/val/(optional) test :pyclass:`torch.utils.data.DataLoader`s.

    Parameters
    ----------
    cfg : RunConfig
        The run configuration containing dataset and model specifications.
    data_config : DatasetConfig | None, optional
        If provided, overrides the dataset configuration in `cfg`. If `None`, uses
        the dataset configuration specified in `cfg.dataset_config`.
    device : str, optional
        The device to use for the DataLoader workers. Defaults to "cpu". If set to

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader | None]
        ``(train_dl, val_dl, test_dl)`` where *test_dl* may be ``None`` if the
        dataset bundle does not include a test split.

    Raises
    ------
    ValueError
        If the dataset does not have more than one label for classification tasks.
    """

    # CUDA requires "spawn" start method; safe on CPU too
    if device != "cpu":
        multiprocessing.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------ #
    # Dataset configuration
    # ------------------------------------------------------------------ #
    if data_config is None:
        train_data_config = load_config(cfg.dataset_config["train"], config_type="data")
        val_data_config = load_config(cfg.dataset_config["valid"], config_type="data")
        test_data_config = None
        if "test" in cfg.dataset_config:
            test_data_config = load_config(
                cfg.dataset_config["test"], config_type="data"
            )
    else:
        train_data_config = data_config
        val_data_config = data_config
        test_data_config = None

    # ------------------------------------------------------------------ #
    # Augmentations & post-processing
    # ------------------------------------------------------------------ #
    postprocessors: list[Any] = []
    aug_processor: Optional[AugmentationProcessor] = None
    if cfg.augmentations:
        aug_device = "cpu"  # DataLoader workers run on CPU
        aug_processor = AugmentationProcessor(cfg.augmentations, cfg.sr, aug_device)
        postprocessors.append(make_item_postprocessor(aug_processor))

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    ds_train, train_metadata = dataset_from_config(train_data_config)
    ds_train = AudioDataset(ds_train, postprocessors=postprocessors)

    ds_val, _ = dataset_from_config(val_data_config)

    # Apply label_to_feature transform to validation set
    if "label_from_feature" or "labels_from_features" in train_metadata:
        label_map = train_metadata["label_from_feature"].get("label_map", {})
        label_transform = LabelFromFeatureConfig(
            feature=train_metadata["label_from_feature"]["label_feature"],
            output_feature="label",
            label_map=label_map,
        )
        ds_val.apply_transformations([label_transform])
    ds_val = AudioDataset(ds_val, postprocessors=postprocessors)

    # Test set may not exist in every dataset bundle
    ds_test = None
    if test_data_config is not None:
        ds_test, _ = dataset_from_config(test_data_config)
        ds_test.apply_transformations([label_transform])
        ds_test = AudioDataset(ds_test, postprocessors=postprocessors)

    num_labels = train_metadata["label_from_feature"].get("num_labels", 0)
    if num_labels <= 1:
        raise ValueError(
            "Dataset must have more than one label for classification tasks."
        )

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(ds_train)
        val_sampler = DistributedSampler(ds_val, shuffle=False)

    # ------------------------------------------------------------------ #
    # Collaters
    # ------------------------------------------------------------------ #
    collate_fn_train = Collater(
        audio_max_length_seconds=cfg.model_spec.audio_config.target_length_seconds,
        sr=cfg.model_spec.audio_config.sample_rate,
        window_selection=cfg.model_spec.audio_config.window_selection,
        keep_text=(cfg.label_type == "text"),
        device=device,
        batch_aug_processor=aug_processor,
        num_labels=num_labels,
    )
    collate_fn_eval = Collater(
        audio_max_length_seconds=cfg.model_spec.audio_config.target_length_seconds,
        sr=cfg.model_spec.audio_config.sample_rate,
        window_selection=cfg.model_spec.audio_config.window_selection,
        keep_text=(cfg.label_type == "text"),
        device=device,
        batch_aug_processor=None,  # no augmentation during eval
        num_labels=num_labels,
    )

    # ------------------------------------------------------------------ #
    # Persist worker init data for deterministic seeding
    # ------------------------------------------------------------------ #
    global _worker_init_data
    _worker_init_data = {"seed": cfg.seed, "aug_processor": aug_processor}

    # ------------------------------------------------------------------ #
    # DataLoaders
    # ------------------------------------------------------------------ #
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
        drop_last=True,
    )
    val_dl = DataLoader(
        ds_val,
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
    if ds_test is not None:
        test_dl = DataLoader(
            ds_test,
            batch_size=cfg.training_params.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn_eval,  # eval collater (no aug)
            pin_memory=(device != "cpu"),
        )
    else:
        test_dl = None

    return train_dl, val_dl, test_dl
