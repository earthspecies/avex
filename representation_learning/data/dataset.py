"""Dataset classes and utilities for representation learning.

This module provides dataset classes and utilities for loading, processing,
and managing datasets used in representation learning tasks.
"""

from __future__ import annotations

import logging
import multiprocessing
import random
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from esp_data import (
    ConcatenatedDataset,
    Dataset,
    DatasetConfig,
    dataset_from_config,
)

# Temporary patch for AnimalSpeak for compatibility
# with dataset concatenation while relevant issue is raised in esp-data.
from representation_learning.data.animalspeak_column_patch import (
    apply_animalspeak_column_patch,
)

# apply_cloudpathlib_patch()
apply_animalspeak_column_patch()


from torch.utils.data import DataLoader, DistributedSampler  # noqa: E402

from representation_learning.configs import RunConfig  # noqa: E402
from representation_learning.data.audio_utils import (  # noqa: E402
    pad_or_window,  # type: ignore
)
from representation_learning.data.augmentations import (  # noqa: E402
    AugmentationProcessor,
    make_item_postprocessor,
)
from representation_learning.data.configs import DatasetCollectionConfig  # noqa: E402

logger = logging.getLogger(__name__)


class AudioDataset(torch.utils.data.Dataset):
    """A wrapper around a Dataset instance for audio data.
    Allows for post-processing of audio samples after retrieval.
    """

    def __init__(
        self,
        ds: Dataset,
        metadata: dict,
        postprocessors: Optional[list[Any]] = None,
    ) -> None:
        """Initialize the AudioDataset with a Dataset instance."""
        self.ds = ds
        self.metadata = metadata
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


def _build_one_dataset_split(
    cfg_list: list[DatasetConfig],
    concatenate: bool = False,
    concatenate_method: str = "soft",
) -> tuple[Dataset | None, dict[str, Any] | None]:
    """Build a single dataset split from a list of DatasetConfig objects.

    Parameters
    ----------
    cfg_list : list[DatasetConfig]
        List of dataset configurations to build.

    concatenate : bool, optional
        If True, concatenate all datasets in the list into a single dataset.
        Defaults to False.
    concatenate_method : str, optional
        Method to use for concatenation. Options are "soft", "overlap", "hard".
        Defaults to "soft".

    Returns
    -------
    tuple[Dataset, dict[str, Any]] | None
        A tuple containing the concatenated dataset and metadata if
        `concatenate` is True,
        or a single dataset and its metadata if `concatenate` is False.
        Returns None if `cfg_list` is empty.

    Raises
    ------
    ValueError
        If the dataset loading fails, e.g., due to invalid configurations.
    """
    if not cfg_list:
        return None, None

    try:
        ds_list = []
        for ds_cfg in cfg_list:
            # Check if data_root is specified but doesn't exist
            if hasattr(ds_cfg, "data_root") and ds_cfg.data_root:
                data_root_path = Path(str(ds_cfg.data_root))
                if not data_root_path.exists():
                    logger.warning(
                        f"data_root '{ds_cfg.data_root}' not found for dataset '{ds_cfg.dataset_name}'. "
                        f"Using default cloud path."
                    )
                    ds_cfg.data_root = None

            ds, metadata = dataset_from_config(ds_cfg)
            ds_list.append((ds, metadata))

        if concatenate and len(ds_list) > 1:
            # Concatenate all training datasets into one
            ds = ConcatenatedDataset(
                [d[0] for d in ds_list], merge_level=concatenate_method
            )

            return ds, ds_list[0][1]  # return first metadata as representative

        # TODO : Handle case where concatenate is False so you return a list of datasets
        else:
            return ds_list[0]
    except Exception as e:
        raise ValueError(
            f"Failed to load training datasets.Error: {e}\nCheck your dataset configurations."
        ) from e


def _build_datasets(
    cfg: DatasetCollectionConfig,
    postprocessors: list[Callable],
    label_type: str,
) -> list[AudioDataset]:
    """Build datasets from the provided configuration.
    Parameters
    ----------
    cfg : DatasetCollectionConfig
        The configuration containing dataset specifications.

    postprocessors : list[Callable]
        List of postprocessors to apply to the datasets.

    Returns
    -------
    """
    # Build train
    train_ds, train_metadata = _build_one_dataset_split(
        cfg.train_datasets, cfg.concatenate_train, cfg.concatenate_method
    )

    if cfg.transformations:
        additional_metadata = train_ds.apply_transformations(cfg.transformations)
        if additional_metadata:
            train_metadata = train_metadata or {}
            train_metadata.update(additional_metadata)

    # Build validation
    val_ds, _ = _build_one_dataset_split(
        cfg.val_datasets, cfg.concatenate_val, cfg.concatenate_method
    )

    # Build test
    test_ds, _ = _build_one_dataset_split(
        cfg.test_datasets, cfg.concatenate_test, cfg.concatenate_method
    )

    # Extract label information from transform metadata
    label_map = {}
    num_classes = 0

    # Handle multi-label case - check for labels_from_features transform metadata
    if "labels_from_features" in train_metadata:
        label_transform_metadata = train_metadata["labels_from_features"]
        label_map = label_transform_metadata.get("label_map", {})
        if (
            "num_classes" in label_transform_metadata
            and label_transform_metadata["num_classes"] > 0
        ):
            num_classes = label_transform_metadata["num_classes"]
        else:
            num_classes = len(label_map)

        # Apply same transformations to val/test datasets to ensure consistency
        if val_ds and cfg.transformations:
            val_ds.apply_transformations(cfg.transformations)
        if test_ds and cfg.transformations:
            test_ds.apply_transformations(cfg.transformations)

    # Handle single-label case - check for label_from_feature transform metadata
    elif "label_from_feature" in train_metadata:
        label_transform_metadata = train_metadata["label_from_feature"]
        label_map = label_transform_metadata.get("label_map", {})
        if (
            "num_classes" in label_transform_metadata
            and label_transform_metadata["num_classes"] > 0
        ):
            num_classes = label_transform_metadata["num_classes"]
        else:
            num_classes = len(label_map)

        # Apply same transformations to val/test datasets to ensure consistency
        if val_ds and cfg.transformations:
            val_ds.apply_transformations(cfg.transformations)
        if test_ds and cfg.transformations:
            test_ds.apply_transformations(cfg.transformations)

    train_ds = AudioDataset(
        train_ds,
        metadata={
            "label_map": label_map,
            "num_labels": num_classes,
        },
        postprocessors=postprocessors,
    )

    if val_ds:
        val_ds = AudioDataset(
            val_ds,
            postprocessors=postprocessors,
            metadata={
                "label_map": label_map,
                "num_labels": num_classes,
            },
        )

    if test_ds:
        test_ds = AudioDataset(
            test_ds,
            postprocessors=postprocessors,
            metadata={
                "label_map": label_map,
                "num_labels": num_classes,
            },
        )

    return train_ds, val_ds, test_ds


# --------------------------------------------------------------------------- #
#  Collater
# --------------------------------------------------------------------------- #
class Collater:
    """
    Combines samples into a batch, ensuring every audio clip has the same
    length (`audio_max_length`) by truncating or zero-padding as needed.

    Supports two-step processing:
    1. First truncate to dataset_audio_max_length_seconds (benchmark constraint)
       if specified
    2. Then pad/truncate to audio_max_length_seconds (model requirement)
    """

    def __init__(
        self,
        audio_max_length_seconds: int,
        sr: int,
        window_selection: str = "random",
        preprocessor: Optional[str] = None,
        device: str = "cpu",
        batch_aug_processor: Optional[AugmentationProcessor] = None,
        num_labels: int = 0,
        dataset_audio_max_length_seconds: Optional[int] = None,
    ) -> None:
        self.audio_max_length_seconds = audio_max_length_seconds
        self.dataset_audio_max_length_seconds = dataset_audio_max_length_seconds
        self.window_selection = window_selection
        self.preprocessor = preprocessor
        self.sr = sr
        self.device = device
        self.batch_aug_processor = batch_aug_processor
        self.num_labels = num_labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # First prepare data with uniform lengths
        audios, masks, labels, text_labels = [], [], [], []

        for item in batch:
            # Use "audio" key which is the standard in esp_data,
            # fallback to "raw_wav" for compatibility
            audio_key = "audio" if "audio" in item else "raw_wav"
            wav = torch.as_tensor(item[audio_key])  # (T,) or (C, T)

            # Handle rare corrupted audio without crashing
            if torch.isnan(wav).any() or torch.isinf(wav).any():
                logger.warning(
                    f"Corrupted audio detected (NaN/Inf), replacing with zeros. Shape: {wav.shape}"
                )
                wav = torch.zeros_like(wav)

            # Handle multichannel audio by taking mean across channels
            if wav.dim() == 2:  # (C, T) multichannel format
                wav = wav.mean(dim=0)  # Convert to mono by averaging channels

            # Step 1: Apply dataset constraint (benchmark limit) if specified
            if self.dataset_audio_max_length_seconds is not None:
                dataset_max_samples = self.dataset_audio_max_length_seconds * self.sr
                if wav.size(-1) > dataset_max_samples:
                    # Apply dataset constraint truncation with same window
                    # selection strategy
                    wav, _ = pad_or_window(
                        wav, dataset_max_samples, self.window_selection
                    )

            # Step 2: Apply model requirement (pad/truncate to target length)
            wav, pad_mask = pad_or_window(
                wav,
                self.audio_max_length_seconds * self.sr,
                self.window_selection,
            )
            audios.append(wav)
            masks.append(pad_mask)
            # Some pipelines (e.g. CLAP) do not create numeric labels.  Keep the
            # *length* of the labels list aligned with the batch by inserting a
            # dummy placeholder when the key is missing.
            if "label" in item:
                labels.append(item["label"])
            else:
                labels.append(0)  # Placeholder – will turn into dummy tensor later
            if "text_label" in item:
                txt_lbl = item["text_label"]
                if isinstance(txt_lbl, list) and len(txt_lbl) > 0:
                    txt_lbl = random.choice(txt_lbl)
                text_labels.append(txt_lbl)

        # ------------------------------------
        # Stack into tensors (audio + mask)
        # ------------------------------------
        audio_tensor = torch.stack(audios)  # [B, T] float32
        mask_tensor = torch.stack(masks)  # [B, T] bool

        # Handle different label formats (int → long, multi-hot → float32)
        if all(isinstance(lbl, (int, np.integer)) for lbl in labels):
            # Convert integer labels to one-hot vectors for classification
            if self.num_labels > 0:
                label_tensor = torch.nn.functional.one_hot(
                    torch.tensor(labels, dtype=torch.long),
                    num_classes=self.num_labels,
                ).float()
            else:
                # For CLAP models without numeric labels, create dummy tensor
                label_tensor = torch.zeros((len(labels), 1), dtype=torch.float32)
        else:
            # For multi-label case, convert lists of class indices to one-hot vectors
            label_tensors = []
            for lbl in labels:
                # Create zero tensor of size num_labels
                if self.num_labels > 0:
                    one_hot = torch.zeros(self.num_labels, dtype=torch.float32)
                    # Convert list of indices to tensor and set 1s at those indices
                    if isinstance(lbl, torch.Tensor):
                        indices = lbl.clone().detach().long()
                    else:
                        indices = torch.tensor(lbl, dtype=torch.long)
                    # Ensure indices are within valid range
                    valid_indices = indices[indices < self.num_labels]
                    if len(valid_indices) > 0:
                        one_hot[valid_indices] = 1.0

                    label_tensors.append(one_hot)
                else:
                    # For CLAP models without numeric labels, create dummy tensor
                    label_tensors.append(torch.zeros(1, dtype=torch.float32))
            label_tensor = torch.stack(label_tensors)

        # Apply batch-level mixup AFTER converting labels to consistent tensors
        if self.batch_aug_processor is not None and audios:
            # Create temporary batch with uniform-sized tensors
            temp_batch = [
                {
                    "audio": audio_tensor[i],  # Use standard "audio" key
                    "label": label_tensor[i],
                    "text_label": text_labels[i] if text_labels else None,
                }
                for i in range(len(audios))
            ]
            # Now mixup can safely operate on same-sized tensors
            mixed_batch = self.batch_aug_processor.apply_batch_augmentations(temp_batch)
            # Extract back to tensors
            audio_tensor = torch.stack([item["audio"] for item in mixed_batch])
            label_tensor = torch.stack([item["label"] for item in mixed_batch])
            text_labels = [item.get("text_label") for item in mixed_batch]

        return {
            # Keep raw_wav for backward compatibility with models
            "raw_wav": audio_tensor,
            "padding_mask": mask_tensor,
            "label": label_tensor,
            "text_label": text_labels,
        }


# --------------------------------------------------------------------------- #
#  Data-loader helpers
# --------------------------------------------------------------------------- #
def worker_init_fn(worker_id: int) -> None:
    """Initialize a DataLoader worker (seeding for reproducibility)."""
    import random

    import numpy as np
    import torch

    # Get the seed that PyTorch already set for this worker
    worker_seed = torch.initial_seed() % 2**32

    # Seed other libraries to ensure reproducibility
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
#  Main builder
# --------------------------------------------------------------------------- #
def build_dataloaders(
    cfg: RunConfig,
    *,
    data_config: DatasetCollectionConfig | None = None,
    device: str = "cpu",
    task_type: str | None = None,
    dataset_audio_max_length_seconds: Optional[int] = None,
    enable_eval_augmentations: bool = False,
    is_evaluation_context: bool = False,
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

    task_type : str | None, optional
        The task type ("detection", "classification", etc.). If provided, allows
        single-class datasets for detection tasks. Defaults to None.

    dataset_audio_max_length_seconds : Optional[int], optional
        Maximum audio length in seconds to use from the source audio before
        applying model target length. This represents benchmark constraints.
        If None, no dataset-level constraint is applied. Defaults to None.

    enable_eval_augmentations : bool, optional
        Whether to enable augmentations during evaluation. Defaults to False.
        This is needed for BirdSet linear probing where train/test are different
        datasets.

    is_evaluation_context : bool, optional
        Whether we're in evaluation context (run_evaluate.py) vs training context
        (run_train.py). In evaluation context, augmentations are disabled by default
        for train/val unless enable_eval_augmentations=True. Defaults to False.

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

    # CUDA requires "spawn" start method; safe on CPU too - do earlier
    # if device != "cpu":
    #     multiprocessing.set_start_method("spawn", force=True)

    postprocessors: list[Any] = []
    aug_processor: Optional[AugmentationProcessor] = None
    if cfg.augmentations:
        aug_device = "cpu"  # DataLoader workers run on CPU
        aug_processor = AugmentationProcessor(cfg.augmentations, cfg.sr, aug_device)
        postprocessors.append(make_item_postprocessor(aug_processor))

    # Build datasets from the configuration
    if data_config is None:
        data_config = cfg.dataset_config

    ds_train, ds_val, ds_test = _build_datasets(
        data_config, postprocessors=postprocessors, label_type=cfg.label_type
    )

    num_labels = ds_train.metadata.get("num_labels", 0)

    # Allow single-class datasets for detection tasks, but require multiple
    # classes for classification
    if cfg.label_type == "supervised":
        if task_type == "detection":
            # For detection tasks, we need at least 1 class
            # (binary detection: present/absent)
            if num_labels < 1:
                raise ValueError("Detection tasks must have at least one label.")
        else:
            # For classification tasks (or unknown task types), require multiple classes
            if num_labels <= 1:
                raise ValueError(
                    "Dataset must have more than one label for classification tasks."
                )
    # For CLAP/CLIP models (label_type="text"), numeric labels are not required

    logger.info(f"Train data size : {len(ds_train)} samples")
    logger.info(f"Validation data size : {len(ds_val)} samples")
    if ds_test is not None:
        logger.info(f"Test data size : {len(ds_test)} samples")

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(ds_train)
        val_sampler = DistributedSampler(ds_val, shuffle=False)

    # ------------------------------------------------------------------ #
    # Collaters
    # ------------------------------------------------------------------ #

    # Determine augmentation strategy based on context
    if is_evaluation_context:
        # Evaluation context (run_evaluate.py):
        # - Default: no augmentations for train/val
        # - BirdSet: augmentations for train/val (enable_eval_augmentations=True)
        train_aug_processor = aug_processor if enable_eval_augmentations else None
        eval_aug_processor = aug_processor if enable_eval_augmentations else None
    else:
        # Training context (run_train.py):
        # - Train: augmentations enabled (standard training behavior)
        # - Val: no augmentations (standard validation behavior)
        train_aug_processor = aug_processor
        eval_aug_processor = None

    # Test augmentations are ALWAYS off, regardless of context or dataset type
    test_aug_processor = None

    collate_fn_train = Collater(
        audio_max_length_seconds=cfg.model_spec.audio_config.target_length_seconds,
        sr=cfg.model_spec.audio_config.sample_rate,
        window_selection=cfg.model_spec.audio_config.window_selection,
        device=device,
        batch_aug_processor=train_aug_processor,
        num_labels=num_labels,
        dataset_audio_max_length_seconds=dataset_audio_max_length_seconds,
    )
    collate_fn_eval = Collater(
        audio_max_length_seconds=cfg.model_spec.audio_config.target_length_seconds,
        sr=cfg.model_spec.audio_config.sample_rate,
        window_selection=cfg.model_spec.audio_config.window_selection,
        device=device,
        batch_aug_processor=eval_aug_processor,
        num_labels=num_labels,
        dataset_audio_max_length_seconds=dataset_audio_max_length_seconds,
    )
    collate_fn_test = Collater(
        audio_max_length_seconds=cfg.model_spec.audio_config.target_length_seconds,
        sr=cfg.model_spec.audio_config.sample_rate,
        window_selection=cfg.model_spec.audio_config.window_selection,
        device=device,
        batch_aug_processor=test_aug_processor,  # ALWAYS None
        num_labels=num_labels,
        dataset_audio_max_length_seconds=dataset_audio_max_length_seconds,
    )

    # ------------------------------------------------------------------ #
    # DataLoaders with proper seeding
    # ------------------------------------------------------------------ #
    # Create generator for reproducible worker seeding
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    # Force "spawn" start method even if the global context is already locked
    ctx = multiprocessing.get_context("spawn")

    train_dl = DataLoader(
        ds_train,
        batch_size=cfg.training_params.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_train,
        pin_memory=(device != "cpu"),
        worker_init_fn=worker_init_fn,
        generator=g,
        multiprocessing_context=ctx,
        persistent_workers=(cfg.num_workers > 0),
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
        generator=g,
        multiprocessing_context=ctx,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    if ds_test is not None:
        test_dl = DataLoader(
            ds_test,
            batch_size=cfg.training_params.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn_test,  # test collater (NEVER has augmentations)
            pin_memory=(device != "cpu"),
            worker_init_fn=worker_init_fn,
            generator=g,
            multiprocessing_context=ctx,
        )
    else:
        test_dl = None

    return train_dl, val_dl, test_dl
