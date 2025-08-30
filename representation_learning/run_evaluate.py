"""
Entry-point script for linear-probe / fine-tuning experiments.

Key points
----------
* **No duplicate forward-pass**: train, val **and test** embeddings are computed
  once and re-used for retrieval evaluation.
* Linear-probe test accuracy is still measured **end-to-end on raw audio** so it
  reflects real inference cost.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Seems to prevent a cloudpathlib error
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "okapi-274503")

import pandas as pd
import torch
from esp_data import DatasetConfig
from esp_data.io import anypath

# Import representation_learning modules
from representation_learning.configs import (
    DatasetCollectionConfig,
    EvaluateConfig,
    EvaluationSet,
    ExperimentConfig,
)
from representation_learning.data.dataset import build_dataloaders
from representation_learning.evaluation.clustering import (
    eval_clustering,
)
from representation_learning.evaluation.embedding_utils import (
    EmbeddingDataset,
    extract_embeddings_for_split,
    load_embeddings_arrays,
    save_embeddings_arrays,
)
from representation_learning.evaluation.finetune import (
    train_and_eval_offline,
    train_and_eval_online,
)
from representation_learning.evaluation.retrieval import (
    eval_retrieval,
    eval_retrieval_cross_set,
)
from representation_learning.models.get_model import get_model
from representation_learning.utils import ExperimentLogger
from representation_learning.utils.experiment_tracking import (
    create_experiment_summary_csvs,
    get_or_create_experiment_metadata,
    save_evaluation_metadata,
)
from representation_learning.utils.utils import _process_state_dict

logger = logging.getLogger("run_finetune")


# -------------------------------------------------------------------- #
#  Dataclass to collect final results
# -------------------------------------------------------------------- #
@dataclass
class ExperimentResult:
    dataset_name: str
    experiment_name: str
    evaluation_dataset_name: Optional[
        str
    ]  # The evaluation set name (e.g., "giant_otters_vocalization")
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    probe_test_metrics: Dict[str, float]
    retrieval_metrics: Dict[str, float]
    clustering_metrics: Dict[str, float]


@dataclass
class ExperimentResultWithModel:
    """Result that includes model caching information for optimization."""

    result: ExperimentResult
    model: Optional[torch.nn.Module]
    model_metadata: Optional[Dict[str, str]]


# -------------------------------------------------------------------- #
#  Core routine for one (dataset, experiment) pair
# -------------------------------------------------------------------- #
def run_experiment(
    eval_cfg: EvaluateConfig,
    dataset_cfg: DatasetConfig,
    experiment_cfg: ExperimentConfig,
    data_collection_cfg: DatasetCollectionConfig,
    device: torch.device,
    save_dir: Path,
    evaluation_dataset_name: Optional[str] = None,
    evaluation_set_metrics: Optional[List[str]] = None,
    evaluation_set: Optional[EvaluationSet] = None,
    cached_model: Optional[torch.nn.Module] = None,
    cached_model_metadata: Optional[Dict[str, str]] = None,
) -> ExperimentResultWithModel:
    logger.info("running experiment")
    dataset_name = dataset_cfg.dataset_name
    experiment_name = experiment_cfg.run_name
    frozen = experiment_cfg.is_frozen()

    logger.info(
        "Running experiment '%s' on dataset '%s' with frozen: %s",
        experiment_name,
        dataset_name,
        frozen,
    )

    # ------------------------------------------------------------------ #
    #  Build run config
    # ------------------------------------------------------------------ #
    run_cfg = experiment_cfg.run_config
    run_cfg.model_spec.audio_config.window_selection = "start"
    run_cfg.training_params = eval_cfg.training_params
    run_cfg.model_spec.device = str(device)

    # Check for BirdSet dataset early to preserve augmentations if needed
    is_birdset = (
        hasattr(dataset_cfg, "dataset_name")
        and "birdset" in dataset_cfg.dataset_name.lower()
    )

    if not is_birdset:
        run_cfg.augmentations = []  # disable training-time augs during (most) eval

    if run_cfg.model_spec.audio_config.sample_rate is not None:
        dataset_cfg.sample_rate = run_cfg.model_spec.audio_config.sample_rate
    else:
        run_cfg.model_spec.audio_config.sample_rate = dataset_cfg.sample_rate
        logger.info(f"Using benchmark sample rate: {dataset_cfg.sample_rate} Hz")

    # Force data collection config to use the model sample rate
    # TODO: just one place
    target_sample_rate = dataset_cfg.sample_rate
    for dataset_list in [
        data_collection_cfg.train_datasets,
        data_collection_cfg.val_datasets,
        data_collection_cfg.test_datasets,
    ]:
        if dataset_list:
            for ds_cfg in dataset_list:
                if ds_cfg.sample_rate != target_sample_rate:
                    ds_cfg.sample_rate = target_sample_rate
                    logging.warning(
                        f"Overriding sample rate for {ds_cfg.dataset_name} "
                        f"to model sample rate {target_sample_rate}"
                    )

    embedding_dir_name = evaluation_dataset_name or dataset_name
    emb_base_dir = save_dir / experiment_name / embedding_dir_name
    train_path = emb_base_dir / "embedding_train.h5"
    val_path = emb_base_dir / "embedding_val.h5"
    test_path = emb_base_dir / "embedding_test.h5"

    # Flags to determine which embeddings we need
    need_probe = "probe" in eval_cfg.eval_modes
    need_retrieval = "retrieval" in eval_cfg.eval_modes
    need_clustering = "clustering" in eval_cfg.eval_modes

    # Get training mode and aggregation method from probe configuration
    # Note: online_training and offline_training are mutually exclusive by design
    online_training = need_probe and experiment_cfg.get_training_mode()
    offline_training = need_probe and not experiment_cfg.get_training_mode()
    aggregation_method = experiment_cfg.get_aggregation_method()

    # For online training (LSTM probes), we don't need to extract embeddings for
    # training
    # For offline training (linear probes), we extract embeddings for training
    need_embedding_extraction_train = (
        need_probe or need_retrieval or need_clustering
    ) and not online_training

    # For retrieval and clustering, we always need test embeddings
    # Use "mean" aggregation for evaluation even when training online with LSTM
    need_embedding_extraction_test = need_retrieval or need_clustering

    logger.info(
        f"Need to write embeddings for train: {need_embedding_extraction_train} "
        f"and test: {need_embedding_extraction_test}"
    )

    if online_training:
        logger.info("Training online.")
    elif offline_training:
        logger.info("Training offline.")
    else:
        logger.info("Not training.")

    logger.info(f"Need to probe: {need_probe}")
    logger.info(f"Need to retrieve: {need_retrieval}")
    logger.info(f"Need to cluster: {need_clustering}")

    # Determine retrieval mode for this evaluation set
    retrieval_mode = (
        evaluation_set.retrieval_mode
        if evaluation_set and evaluation_set.retrieval_mode is not None
        else "test_vs_test"
    )

    overwrite = getattr(eval_cfg, "overwrite_embeddings", False)

    # Determine what embeddings need to be recomputed
    if overwrite:
        # When overwrite=True, only recompute what we actually need
        need_recompute_embeddings_train = need_embedding_extraction_train
        need_recompute_embeddings_test = need_embedding_extraction_test

        if need_recompute_embeddings_train or need_recompute_embeddings_test:
            logger.info(
                f"Forcing overwrite of embeddings: "
                f"train={need_recompute_embeddings_train}, "
                f"test={need_recompute_embeddings_test} due to "
                f"overwrite_embeddings=True"
            )

            # Remove existing files that we're going to recompute
            paths_to_remove = []
            if need_recompute_embeddings_train:
                paths_to_remove.extend([train_path, val_path])
            if need_recompute_embeddings_test:
                paths_to_remove.append(test_path)

            for path in paths_to_remove:
                if path.exists():
                    try:
                        path.unlink()
                        logger.info(f"Removed existing embedding file: {path}")
                    except Exception as e:
                        logger.warning(f"Could not remove {path}: {e}")
    else:
        # Normal logic: check file existence
        need_recompute_embeddings_train = (
            need_probe and not (train_path.exists() and val_path.exists())
        ) or (
            need_retrieval
            and retrieval_mode == "train_vs_test"
            and not train_path.exists()
        )
        need_recompute_embeddings_test = (
            need_retrieval or need_probe or need_clustering
        ) and not test_path.exists()
    logger.info(
        "Need to recompute embeddings for train: %s and test: %s",
        need_recompute_embeddings_train,
        need_recompute_embeddings_test,
    )

    need_base_model = online_training or need_recompute_embeddings_train
    logger.info(f"Need to load base model: {need_base_model}")

    need_raw_dataloaders = online_training or need_recompute_embeddings_train
    logger.info(f"Need to build raw dataloaders: {need_raw_dataloaders}")

    # ------------------------------------------------------------------ #
    #  Dataloaders (raw audio)
    # ------------------------------------------------------------------ #
    train_dl_raw = None
    val_dl_raw = None
    test_dl_raw = None
    num_labels = None

    if need_raw_dataloaders:
        eval_data_collection_cfg = DatasetCollectionConfig(
            train_datasets=data_collection_cfg.train_datasets
            or ([dataset_cfg] if "probe" in eval_cfg.eval_modes else None),
            val_datasets=data_collection_cfg.val_datasets
            or ([dataset_cfg] if "probe" in eval_cfg.eval_modes else None),
            test_datasets=[dataset_cfg],  # Always use the current dataset for testing
        )

        # Extract dataset-level audio constraint (benchmark constraint) if it exists
        dataset_audio_max_length = getattr(
            dataset_cfg, "audio_max_length_seconds", None
        )
        if dataset_audio_max_length is not None:
            logger.info(
                f"Dataset audio constraint: {dataset_audio_max_length}s, "
                f"Model target length: "
                f"{run_cfg.model_spec.audio_config.target_length_seconds}s"
            )

        # For BirdSet datasets, augment during eval (detection done earlier)
        if is_birdset:
            logger.info(
                f"BirdSet dataset detected ({dataset_name}). "
                "Enabling augmentations for train/validation during evaluation "
                "(test always has no augmentations)."
            )

        train_dl_raw, val_dl_raw, test_dl_raw = build_dataloaders(
            run_cfg,
            data_config=eval_data_collection_cfg,
            device=str(device),
            task_type=getattr(dataset_cfg, "type", None),
            dataset_audio_max_length_seconds=dataset_audio_max_length,
            enable_eval_augmentations=is_birdset,
            is_evaluation_context=True,
        )

        # Log dataloader info
        train_batches = len(train_dl_raw) if train_dl_raw else 0
        val_batches = len(val_dl_raw) if val_dl_raw else 0
        test_batches = len(test_dl_raw) if test_dl_raw else 0

        logger.info(
            "Raw dataloaders ready: %d/%d/%d raw batches",
            train_batches,
            val_batches,
            test_batches,
        )

        if test_dl_raw and hasattr(test_dl_raw.dataset, "metadata"):
            num_labels = test_dl_raw.dataset.metadata.get("num_labels", None)

    # ------------------------------------------------------------------ #
    #  Backbone (optionally load checkpoint)
    # ------------------------------------------------------------------ #
    base_model: torch.nn.Module | None = None

    if need_base_model:
        if num_labels is None:
            raise ValueError(
                "Could not determine number of labels from "
                "embeddings or raw dataloaders"
            )

        # Check if we can reuse cached model
        experiment_cfg.checkpoint_path = experiment_cfg.checkpoint_path

        if not frozen:
            logger.info("Loading fresh model for fine-tuning (frozen=False)")
            base_model = get_model(run_cfg.model_spec, num_classes=num_labels).to(
                device
            )
        elif (
            cached_model is not None
            and cached_model_metadata is not None
            and cached_model_metadata.get("checkpoint_path")
            == experiment_cfg.checkpoint_path
            and cached_model_metadata.get("frozen") == str(frozen)
        ):
            logger.info("Reusing cached model from previous dataset (frozen=True)")
            base_model = cached_model
        else:
            logger.info("Loading model (cache miss or first dataset)")
            base_model = get_model(run_cfg.model_spec, num_classes=num_labels).to(
                device
            )

            if experiment_cfg.checkpoint_path:
                ckpt_path = anypath(experiment_cfg.checkpoint_path)
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                with ckpt_path.open("rb") as f:
                    state = torch.load(f, map_location=device)
                if "model_state_dict" in state:
                    state = _process_state_dict(state)

                base_model.load_state_dict(state, strict=False)
                logger.info("Loaded checkpoint from %s", ckpt_path)

        if frozen:
            base_model.eval()

        # Update cached model metadata for next iteration
        if frozen:
            cached_model_metadata = {
                "checkpoint_path": experiment_cfg.checkpoint_path,
                "frozen": str(frozen),
            }
        else:
            # Don't cache fine-tuning models since weights get modified
            cached_model_metadata = None

    # ------------------------------------------------------------------ #
    #  Layer selection
    # ------------------------------------------------------------------ #
    if base_model is not None:
        layer_names = experiment_cfg.get_target_layers()
    else:
        # When base_model is None, we don't need layer names
        layer_names = []

    # ------------------------------------------------------------------ #
    #  Calculate target_length for probes
    # ------------------------------------------------------------------ #
    target_length = None
    if dataset_audio_max_length is not None:
        # Convert audio_max_length_seconds to samples using the model's sample rate
        target_length = int(
            dataset_audio_max_length * run_cfg.model_spec.audio_config.sample_rate
        )
        logger.info(
            f"Using dataset-specific target_length: {target_length} samples "
            f"({dataset_audio_max_length}s * "
            f"{run_cfg.model_spec.audio_config.sample_rate} Hz)"
        )
    else:
        logger.info(
            "No dataset audio_max_length_seconds specified, using default target_length"
        )

    train_ds: EmbeddingDataset | None = None  # will remain None if not needed
    val_ds: EmbeddingDataset | None = None
    test_embeds: torch.Tensor | None = None
    test_labels: torch.Tensor | None = None
    train_embeds: torch.Tensor | None = None
    train_labels: torch.Tensor | None = None

    # ------------------- embeddings for linear probe ------------------- #
    if offline_training:
        if not need_recompute_embeddings_train:
            train_embeds, train_labels, num_labels = load_embeddings_arrays(train_path)
            val_embeds, val_labels, _ = load_embeddings_arrays(val_path)
        else:
            if base_model is None:
                raise ValueError("base_model is required to compute embeddings")

            # Use streaming approach for memory efficiency when extracting from many
            # layers. This prevents OOM issues when using "all" layers in EfficientNet
            use_streaming = eval_cfg.use_streaming_embeddings and (
                len(layer_names) > 5 or "all" in layer_names
            )

            if use_streaming:
                logger.info(
                    f"Using streaming approach for memory-efficient embedding "
                    f"extraction (layers: {len(layer_names)}, streaming enabled: "
                    f"{eval_cfg.use_streaming_embeddings}) - "
                    f"Computing embeddings for train and val"
                )
                train_embeds, train_labels = extract_embeddings_for_split(
                    base_model,
                    train_dl_raw,
                    layer_names,
                    device,
                    aggregation=aggregation_method,
                    save_path=train_path,
                    chunk_size=eval_cfg.streaming_chunk_size,
                    compression=eval_cfg.hdf5_compression,
                    compression_level=eval_cfg.hdf5_compression_level,
                    auto_chunk_size=eval_cfg.auto_chunk_size,
                    max_chunk_size=eval_cfg.max_chunk_size,
                    min_chunk_size=eval_cfg.min_chunk_size,
                    batch_chunk_size=getattr(eval_cfg, "batch_chunk_size", 10),
                )
                val_embeds, val_labels = extract_embeddings_for_split(
                    base_model,
                    val_dl_raw,
                    layer_names,
                    device,
                    aggregation=aggregation_method,
                    save_path=val_path,
                    chunk_size=eval_cfg.streaming_chunk_size,
                    compression=eval_cfg.hdf5_compression,
                    compression_level=eval_cfg.hdf5_compression_level,
                    auto_chunk_size=eval_cfg.auto_chunk_size,
                    max_chunk_size=eval_cfg.max_chunk_size,
                    min_chunk_size=eval_cfg.min_chunk_size,
                    batch_chunk_size=getattr(eval_cfg, "batch_chunk_size", 10),
                )
            else:
                logger.info(
                    f"Using standard in-memory embedding extraction "
                    f"(layers: {len(layer_names)}, streaming disabled: "
                    f"{eval_cfg.use_streaming_embeddings}) - "
                    f"Computing embeddings for train and val"
                )
                train_embeds, train_labels = extract_embeddings_for_split(
                    base_model,
                    train_dl_raw,
                    layer_names,
                    device,
                    aggregation=aggregation_method,
                    batch_chunk_size=getattr(eval_cfg, "batch_chunk_size", 10),
                )
                val_embeds, val_labels = extract_embeddings_for_split(
                    base_model,
                    val_dl_raw,
                    layer_names,
                    device,
                    aggregation=aggregation_method,
                    batch_chunk_size=getattr(eval_cfg, "batch_chunk_size", 10),
                )

                save_embeddings_arrays(
                    train_embeds,
                    train_labels,
                    train_path,
                    num_labels,
                    compression=eval_cfg.hdf5_compression,
                    compression_level=eval_cfg.hdf5_compression_level,
                )
                save_embeddings_arrays(
                    val_embeds,
                    val_labels,
                    val_path,
                    num_labels,
                    compression=eval_cfg.hdf5_compression,
                    compression_level=eval_cfg.hdf5_compression_level,
                )

        train_ds = EmbeddingDataset(train_embeds, train_labels)
        val_ds = EmbeddingDataset(val_embeds, val_labels)
        num_labels = len(train_labels.unique()) if num_labels is None else num_labels

    # ------------------- embeddings for train-vs-test retrieval -------- #
    if need_retrieval and retrieval_mode == "train_vs_test" and train_embeds is None:
        if not need_recompute_embeddings_train:
            train_embeds, train_labels, _ = load_embeddings_arrays(train_path)
        else:
            if base_model is None:
                raise ValueError("base_model is required to compute embeddings")

            # Use streaming approach for memory efficiency when extracting from many
            # layers
            use_streaming = eval_cfg.use_streaming_embeddings and (
                len(layer_names) > 5 or "all" in layer_names
            )

            # We need to force aggregation method to "mean" for retrieval if it's
            # different from "mean" or "max"
            if aggregation_method not in ["mean", "max"]:
                aggregation_method_retrieval = "mean"
                logger.info(
                    f"Forcing aggregation method to 'mean' for retrieval "
                    f"(aggregation_method: {aggregation_method_retrieval})"
                )
            else:
                aggregation_method_retrieval = aggregation_method

            if use_streaming:
                logger.info(
                    f"Using streaming approach for memory-efficient embedding "
                    f"extraction (retrieval) (layers: {len(layer_names)}, streaming "
                    f"enabled: {eval_cfg.use_streaming_embeddings}) - "
                    f"Computing embeddings for train and val"
                )
                train_embeds, train_labels = extract_embeddings_for_split(
                    base_model,
                    train_dl_raw,
                    layer_names,
                    device,
                    aggregation=aggregation_method_retrieval,
                    save_path=train_path,
                    chunk_size=eval_cfg.streaming_chunk_size,
                    compression=eval_cfg.hdf5_compression,
                    compression_level=eval_cfg.hdf5_compression_level,
                    auto_chunk_size=eval_cfg.auto_chunk_size,
                    max_chunk_size=eval_cfg.max_chunk_size,
                    min_chunk_size=eval_cfg.min_chunk_size,
                    batch_chunk_size=getattr(eval_cfg, "batch_chunk_size", 10),
                )
            else:
                logger.info(
                    f"Using standard in-memory embedding extraction (retrieval) "
                    f"(layers: {len(layer_names)}, streaming disabled: "
                    f"{eval_cfg.use_streaming_embeddings})"
                )
                train_embeds, train_labels = extract_embeddings_for_split(
                    base_model,
                    train_dl_raw,
                    layer_names,
                    device,
                    aggregation=aggregation_method_retrieval,
                    batch_chunk_size=getattr(eval_cfg, "batch_chunk_size", 10),
                )

                save_embeddings_arrays(
                    train_embeds,
                    train_labels,
                    train_path,
                    num_labels,
                    compression=eval_cfg.hdf5_compression,
                    compression_level=eval_cfg.hdf5_compression_level,
                )

    # ------------------- embeddings for retrieval and clustering -------- #
    if need_retrieval or need_probe or need_clustering:
        test_path = emb_base_dir / "embedding_test.h5"
        logger.info(test_path)

        if (not overwrite) and test_path.exists():
            test_embeds, test_labels, _ = load_embeddings_arrays(test_path)
        else:
            if base_model is None:
                raise ValueError("base_model is required to compute embeddings")

            # Use streaming approach for memory efficiency when extracting from many
            # layers
            use_streaming = eval_cfg.use_streaming_embeddings and (
                len(layer_names) > 5 or "all" in layer_names
            )

            # We need to force aggregation method to "mean" for retrieval if it's
            # different from "mean" or "max"
            if aggregation_method not in ["mean", "max"]:
                aggregation_method_retrieval = "mean"
                logger.info(
                    f"Forcing aggregation method to 'mean' for retrieval "
                    f"(aggregation_method: {aggregation_method_retrieval})"
                )
            else:
                aggregation_method_retrieval = aggregation_method

            if use_streaming:
                logger.info(
                    f"Using streaming approach for memory-efficient embedding "
                    f"extraction (test) (layers: {len(layer_names)}, streaming "
                    f"enabled: {eval_cfg.use_streaming_embeddings}) - "
                    f"Computing embeddings for test"
                )
                test_embeds, test_labels = extract_embeddings_for_split(
                    base_model,
                    test_dl_raw,
                    layer_names,
                    device,
                    aggregation=aggregation_method_retrieval,
                    save_path=test_path,
                    chunk_size=eval_cfg.streaming_chunk_size,
                    compression=eval_cfg.hdf5_compression,
                    compression_level=eval_cfg.hdf5_compression_level,
                    auto_chunk_size=eval_cfg.auto_chunk_size,
                    max_chunk_size=eval_cfg.max_chunk_size,
                    min_chunk_size=eval_cfg.min_chunk_size,
                )
            else:
                logger.info(
                    f"Using standard in-memory embedding extraction (test) "
                    f"(layers: {len(layer_names)}, streaming disabled: "
                    f"{eval_cfg.use_streaming_embeddings}) - "
                    f"Computing embeddings for test"
                )
                test_embeds, test_labels = extract_embeddings_for_split(
                    base_model,
                    test_dl_raw,
                    layer_names,
                    device,
                    aggregation=aggregation_method_retrieval,
                    batch_chunk_size=getattr(eval_cfg, "batch_chunk_size", 10),
                )

                save_embeddings_arrays(
                    test_embeds,
                    test_labels,
                    test_path,
                    num_labels,
                    compression=eval_cfg.hdf5_compression,
                    compression_level=eval_cfg.hdf5_compression_level,
                )

        num_labels = len(test_labels.unique()) if num_labels is None else num_labels

    # ------------------------------------------------------------------ #
    #  Experiment logger
    # ------------------------------------------------------------------ #
    exp_logger = ExperimentLogger.from_config(experiment_cfg)
    log_dir_name = evaluation_dataset_name or dataset_name
    exp_logger.log_dir = save_dir / experiment_name / log_dir_name
    exp_logger.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  (1) Linear probe
    # ------------------------------------------------------------------ #
    train_metrics: Dict[str, float] = {}
    val_metrics: Dict[str, float] = {}
    probe_test_metrics: Dict[str, float] = {}

    if "probe" in eval_cfg.eval_modes:
        dataset_metrics = evaluation_set_metrics or getattr(
            dataset_cfg, "metrics", None
        )

        # TODO: metrics per task-group
        classification_metrics = [
            m for m in dataset_metrics if not m.startswith("clustering_")
        ]

        if offline_training:
            logger.info("Training offline")
            # Get multi-label setting from dataset config, with fallback based on type
            is_multi_label = getattr(
                dataset_cfg,
                "multi_label",
                getattr(dataset_cfg, "type", None) == "detection",
            )
            (
                train_metrics,
                val_metrics,
                probe_test_metrics,
            ) = train_and_eval_offline(
                train_ds,
                val_ds,
                EmbeddingDataset(test_embeds, test_labels),
                num_labels,
                layer_names,
                eval_cfg,
                device,
                exp_logger,
                is_multi_label,
                dataset_metrics=classification_metrics,
                experiment_cfg=experiment_cfg,
                target_length=target_length,
            )
        else:
            logger.info("Training online")
            # For fine-tuning, use raw dataloaders
            is_multi_label = getattr(
                dataset_cfg,
                "multi_label",
                getattr(dataset_cfg, "type", None) == "detection",
            )

            (
                train_metrics,
                val_metrics,
                probe_test_metrics,
            ) = train_and_eval_online(
                train_dl_raw,
                val_dl_raw,
                test_dl_raw,
                base_model,
                num_labels,
                layer_names,
                eval_cfg,
                device,
                exp_logger,
                is_multi_label,
                dataset_metrics=classification_metrics,
                experiment_cfg=experiment_cfg,
                target_length=target_length,
            )
    else:
        logger.info("Probe not run because not in eval_modes")

    # ------------------------------------------------------------------ #
    #  (2) Retrieval (from cached test embeddings)
    # ------------------------------------------------------------------ #
    retrieval_metrics: Dict[str, float] = {}
    if "retrieval" in eval_cfg.eval_modes:
        if retrieval_mode == "train_vs_test":
            if train_embeds is None:
                raise ValueError("train_embeds is required for train_vs_test retrieval")
            retrieval_metrics = eval_retrieval_cross_set(
                train_embeds, train_labels, test_embeds, test_labels
            )
        else:
            retrieval_metrics = eval_retrieval(test_embeds, test_labels)

    # ------------------------------------------------------------------ #
    #  (3) Clustering (from cached test embeddings)
    # ------------------------------------------------------------------ #
    clustering_metrics: Dict[str, float] = {}
    if "clustering" in eval_cfg.eval_modes:
        # Only evaluate clustering at the ground-truth K (no best-K sweep)
        clustering_metrics.update(eval_clustering(test_embeds, test_labels))
        logger.info("Clustering metrics: %s", clustering_metrics)

    # Log & finish
    exp_logger.log_metrics(train_metrics, step=0, split="train_final")
    exp_logger.log_metrics(val_metrics, step=0, split="val_final")
    exp_logger.log_metrics(probe_test_metrics, step=0, split="test_probe")
    exp_logger.log_metrics(retrieval_metrics, step=0, split="test_retrieval")
    exp_logger.log_metrics(clustering_metrics, step=0, split="test_clustering")
    exp_logger.finalize()

    # Get or create training metadata
    training_metadata = pd.DataFrame()
    if experiment_cfg.checkpoint_path:
        checkpoint_dir = Path(experiment_cfg.checkpoint_path).parent
        checkpoint_name = Path(experiment_cfg.checkpoint_path).name
        training_metadata = get_or_create_experiment_metadata(
            output_dir=checkpoint_dir,
            config=run_cfg,
            checkpoint_name=checkpoint_name,
        )

    # Save evaluation metadata - Use evaluation_dataset_name to avoid
    # metadata collisions
    metadata_dataset_name = evaluation_dataset_name or dataset_name
    save_evaluation_metadata(
        output_dir=save_dir,
        dataset_name=metadata_dataset_name,
        experiment_name=experiment_name,
        checkpoint_name=(
            Path(experiment_cfg.checkpoint_path).name
            if experiment_cfg.checkpoint_path
            else "None"
        ),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        probe_test_metrics=probe_test_metrics,
        retrieval_metrics=retrieval_metrics,
        clustering_metrics=clustering_metrics,
        eval_config=eval_cfg.model_dump(mode="json"),
        training_metadata=training_metadata,
        run_config=run_cfg.model_dump(mode="json"),  # Always include run config
    )

    return ExperimentResultWithModel(
        result=ExperimentResult(
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            evaluation_dataset_name=evaluation_dataset_name,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            probe_test_metrics=probe_test_metrics,
            retrieval_metrics=retrieval_metrics,
            clustering_metrics=clustering_metrics,
        ),
        model=base_model if frozen else None,  # Only cache frozen models
        model_metadata=cached_model_metadata,
    )


# -------------------------------------------------------------------- #
#  Main
# -------------------------------------------------------------------- #
def main(config_path: Path, patches: tuple[str, ...] | None = None) -> None:
    """
    Main entry point for evaluation.
    """

    # 1. Load configs
    # 1. Load configs with patches
    if patches is None:
        patches = ()
    eval_cfg = EvaluateConfig.from_sources(yaml_file=config_path, cli_args=patches)

    benchmark_eval_cfg = eval_cfg.dataset_config
    evaluation_sets = benchmark_eval_cfg.get_all_evaluation_sets()
    if not evaluation_sets:
        logger.error(
            "No evaluation sets found in BenchmarkEvaluationConfig. "
            "Nothing to evaluate."
        )
        return
    logger.info(f"Loaded {len(evaluation_sets)} evaluation sets")

    # 2. Output dir & device
    if str(eval_cfg.save_dir).startswith("gs://"):
        save_dir = anypath(str(eval_cfg.save_dir))
    else:
        save_dir = Path(str(eval_cfg.save_dir)).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 3. Run experiments - OPTIMIZED: group by experiment to reuse models
    all_results: List[ExperimentResult] = []

    for eval_set_name, _eval_set_data_cfg in evaluation_sets:
        logger.info(f"Evaluating benchmark set: {eval_set_name}")
    if not evaluation_sets:
        logger.warning(
            "No evaluation sets found in BenchmarkEvaluationConfig. "
            "Nothing to evaluate."
        )
        return

    # Group by experiment to load each model only once (saved a lot of time.)
    for exp_cfg in eval_cfg.experiments:
        logger.info(f"Starting experiment: {exp_cfg.run_name}")

        cached_model = None
        model_metadata = None

        for eval_set_name, _eval_set_data_cfg in evaluation_sets:
            logger.info(
                f"Evaluating experiment '{exp_cfg.run_name}' on set: {eval_set_name}"
            )

            # Extract the test dataset from the evaluation set
            test_datasets = _eval_set_data_cfg.test_datasets or []
            if not test_datasets:
                logger.warning(
                    f"No test datasets in evaluation set '{eval_set_name}'. Skipping."
                )
                continue

            # For benchmark evaluation sets, we expect exactly one test dataset per set
            test_ds_cfg = test_datasets[0]

            # Get metrics from the benchmark evaluation config
            eval_set_metrics = benchmark_eval_cfg.get_metrics_for_evaluation_set(
                eval_set_name
            )

            # Get the evaluation set object for configuration
            eval_set = benchmark_eval_cfg.get_evaluation_set(eval_set_name)

            res = run_experiment(
                eval_cfg,
                test_ds_cfg,
                exp_cfg,
                _eval_set_data_cfg,
                device,
                save_dir,
                eval_set_name,
                eval_set_metrics,
                eval_set,
                cached_model,
                model_metadata,
            )
            all_results.append(res.result)

            # Cache the model for reuse in next dataset
            cached_model = res.model
            model_metadata = res.model_metadata

            logger.info(
                "[%s | %s | %s]  probe-test: %s | retrieval: %s | clustering: %s",
                eval_set_name,
                res.result.dataset_name,
                res.result.experiment_name,
                res.result.probe_test_metrics,
                res.result.retrieval_metrics or "n/a",
                res.result.clustering_metrics or "n/a",
            )

    # 5. Create and save summary CSV files
    create_experiment_summary_csvs(
        all_results=all_results,
        eval_cfg=eval_cfg,
        save_dir=save_dir,
        config_file_path=str(config_path),
        benchmark_eval_cfg=benchmark_eval_cfg,
        evaluation_sets=evaluation_sets,
        experiments=eval_cfg.experiments,
    )


if __name__ == "__main__":
    main()
