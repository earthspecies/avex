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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from esp_data import DatasetConfig
from esp_data.io import AnyPathT, anypath, exists
from esp_data.io.paths import PureCloudPath

# Import avex modules
from avex.configs import (
    EvaluateConfig,
    ExperimentConfig,
)
from avex.data.configs import (
    DatasetCollectionConfig,
    EvaluationSet,
)
from avex.data.dataset import build_dataloaders
from avex.evaluation.clustering import (
    eval_clustering,
)
from avex.evaluation.embedding_manager import (
    EmbeddingDataSource,
    EmbeddingDataSourceConfig,
)
from avex.evaluation.embedding_utils import (
    load_embeddings_arrays,
)
from avex.evaluation.finetune import (
    train_and_eval_offline,
    train_and_eval_online,
)
from avex.evaluation.retrieval import (
    eval_retrieval,
    eval_retrieval_cross_set,
)
from avex.models.utils.factory import build_model_from_spec
from avex.utils import ExperimentLogger
from avex.utils.experiment_tracking import (
    create_experiment_summary_csvs,
    get_or_create_experiment_metadata,
    save_evaluation_metadata,
)
from avex.utils.utils import _process_state_dict, universal_torch_load

logger = logging.getLogger("run_finetune")


# -------------------------------------------------------------------- #
#  Dataclass to collect final results
# -------------------------------------------------------------------- #
@dataclass
class ExperimentResult:
    """Result container for experiment evaluation metrics.

    Stores evaluation results from different phases of an experiment including
    training, validation, probing, retrieval, and clustering metrics.
    """

    dataset_name: str
    experiment_name: str
    evaluation_dataset_name: Optional[str]  # The evaluation set name (e.g., "giant_otters_vocalization")
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
    save_dir: Path | AnyPathT,
    evaluation_dataset_name: Optional[str] = None,
    evaluation_set_metrics: Optional[List[str]] = None,
    evaluation_set: Optional[EvaluationSet] = None,
    cached_model: Optional[torch.nn.Module] = None,
    cached_model_metadata: Optional[Dict[str, str]] = None,
) -> ExperimentResultWithModel:
    logger.info("running experiment")
    dataset_name = dataset_cfg.dataset_name
    experiment_name = experiment_cfg.run_name
    freeze_backbone = experiment_cfg.is_frozen()  # Checks probe_config.freeze_backbone

    # Measure total training wall-clock time per evaluation setting
    _training_start_time = time.time()

    logger.info(
        "Running experiment '%s' on dataset '%s' with freeze_backbone: %s",
        experiment_name,
        dataset_name,
        freeze_backbone,
    )

    # ------------------------------------------------------------------ #
    #  Build run config
    # ------------------------------------------------------------------ #
    run_cfg = experiment_cfg.run_config
    run_cfg.model_spec.audio_config.window_selection = "start"
    run_cfg.training_params = eval_cfg.training_params
    run_cfg.model_spec.device = str(device)

    # Check for BirdSet dataset early to preserve augmentations if needed
    is_birdset = hasattr(dataset_cfg, "dataset_name") and "birdset" in dataset_cfg.dataset_name.lower()

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
                        f"Overriding sample rate for {ds_cfg.dataset_name} to model sample rate {target_sample_rate}"
                    )

    # Save embeddings at dataset+model level, not experiment level, so all probe types
    # can reuse the same embeddings for a given dataset+model combination
    embedding_dir_name = evaluation_dataset_name or dataset_name

    # Extract model name from run_config (e.g., "beats_pretrained" from
    # "beats_pretrained.yml")
    model_name = run_cfg.run_name
    if model_name is None:
        # Fallback: try to extract from model_spec.name
        model_name = run_cfg.model_spec.name
        logger.warning(f"No run_name in run_config, using model_spec.name: {model_name}")

    # Create folder structure: {save_dir}/{dataset_name}_{model_name}/
    emb_base_dir = save_dir / f"{embedding_dir_name}_{model_name}"

    # Flags to determine which embeddings we need
    need_probe = "probe" in eval_cfg.eval_modes
    need_retrieval = "retrieval" in eval_cfg.eval_modes
    need_clustering = "clustering" in eval_cfg.eval_modes

    # Get training mode and aggregation method from probe configuration
    online_training = need_probe and experiment_cfg.get_training_mode()

    def generate_embedding_filename(split: str, layer_names: List[str]) -> Path:
        """Generate embedding filename with layer names.

        Args:
            split: Dataset split ('train', 'val', 'test')
            layer_names: List of layer names to extract embeddings from

        Returns:
            Path object for the embedding file
        """
        # Create a safe layer identifier from layer names
        if len(layer_names) == 1:
            # Single layer: use the layer name directly
            layer_id = layer_names[0].replace(".", "_").replace("backbone_", "")
        else:
            # Multiple layers: create a combined identifier
            layer_id = f"multi_{len(layer_names)}_layers"

        # Create filename: embedding_{split}_{layer_id}.h5
        filename = f"embedding_{split}_{layer_id}.h5"
        return emb_base_dir / filename

    # Get layer names for filename generation
    layer_names = experiment_cfg.get_target_layers()

    # Generate filenames based on layer names
    train_path = generate_embedding_filename("train", layer_names)
    val_path = generate_embedding_filename("val", layer_names)
    test_path = generate_embedding_filename("test", layer_names)
    test_path_clustering = emb_base_dir / "embedding_test_clustering.h5"
    train_path_clustering = emb_base_dir / "embedding_train_clustering.h5"

    # Log the generated filenames (only when probing is enabled)
    if need_probe:
        logger.info("Generated embedding filenames:")
        logger.info(f"  Train path: {train_path}")
        logger.info(f"  Val path: {val_path}")
        logger.info(f"  Test path: {test_path}")

    # Log clustering/retrieval paths when those modes are enabled
    if need_retrieval or need_clustering:
        logger.info("Generated clustering/retrieval embedding filenames:")
        logger.info(f"  Train clustering path: {train_path_clustering}")
        logger.info(f"  Test clustering path: {test_path_clustering}")

    # For offline training, always save embeddings with aggregation="none" so they
    # can be reused by different probe types (2D probes can aggregate as needed,
    # 3D probes use full sequence)
    if not online_training:
        aggregation_method = "none"
        logger.info("Using aggregation='none' for offline training to enable probe reuse")
    else:
        aggregation_method = experiment_cfg.get_aggregation_method()

    # Log probe configuration details
    if need_probe:
        probe_type = experiment_cfg.get_probe_type()
        target_layers = experiment_cfg.get_target_layers()
        input_processing = experiment_cfg.get_input_processing_method()
        probe_specific_params = experiment_cfg.get_probe_specific_params()

        logger.info(
            f"Probe configuration: type={probe_type}, "
            f"layers={target_layers}, "
            f"aggregation={aggregation_method}, "
            f"input_processing={input_processing}, "
            f"training_mode={'online' if online_training else 'offline'}"
        )

        if probe_specific_params:
            logger.info(f"Probe-specific parameters: {probe_specific_params}")

    # For online training, we don't need to extract embeddings for training
    # For offline training, we extract embeddings for training
    need_embedding_extraction_probe_train = need_probe and not online_training

    need_embedding_extraction_probe_test = need_probe and not online_training

    logger.info(
        f"Need to write embeddings for probing train: "
        f"{need_embedding_extraction_probe_train} and test: "
        f"{need_embedding_extraction_probe_test}"
    )

    if need_probe:
        if online_training:
            logger.info("Training online.")
        else:
            logger.info("Training offline.")

    logger.info(f"Need to probe: {need_probe}")
    logger.info(f"Need to retrieve: {need_retrieval}")
    logger.info(f"Need to cluster: {need_clustering}")

    # Determine retrieval mode for this evaluation set
    retrieval_mode = (
        evaluation_set.retrieval_mode
        if evaluation_set and evaluation_set.retrieval_mode is not None
        else "test_vs_test"
    )
    need_embedding_extraction_probe_train_clustering = (
        need_clustering or need_retrieval and retrieval_mode == "train_vs_test"
    )
    need_embedding_extraction_probe_test_clustering = need_clustering or need_retrieval

    overwrite = getattr(eval_cfg.offline_embeddings, "overwrite_embeddings", False)

    # Determine what embeddings need to be recomputed
    if overwrite:
        # When overwrite=True, only recompute what we actually need
        need_recompute_embeddings_train = need_embedding_extraction_probe_train
        need_recompute_embeddings_test = need_embedding_extraction_probe_test
        need_recompute_embeddings_train_clustering = need_embedding_extraction_probe_train_clustering
        need_recompute_embeddings_test_clustering = need_embedding_extraction_probe_test_clustering

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
            if need_recompute_embeddings_test_clustering:
                paths_to_remove.append(test_path_clustering)

            for path in paths_to_remove:
                if path.exists():
                    try:
                        path.unlink()
                        logger.info(f"Removed existing embedding file: {path}")
                    except Exception as e:
                        logger.warning(f"Could not remove {path}: {e}")
    else:
        # Normal logic: check file existence for the appropriate aggregation method
        need_recompute_embeddings_train = (
            need_probe and not (train_path.exists() and val_path.exists()) and not online_training
        )
        need_recompute_embeddings_test = need_probe and not online_training and not test_path.exists()
        need_recompute_embeddings_train_clustering = (
            need_clustering
            or need_retrieval
            and retrieval_mode == "train_vs_test"
            and not train_path_clustering.exists()
        )
        need_recompute_embeddings_test_clustering = (
            need_clustering or need_retrieval and not test_path_clustering.exists()
        )
    logger.info(
        "Need to recompute embeddings for probing, train: %s and test: %s",
        need_recompute_embeddings_train,
        need_recompute_embeddings_test,
    )
    logger.info(
        "Need to recompute embeddings for probing, train clustering: %s and test clustering: %s",
        need_recompute_embeddings_train_clustering,
        need_recompute_embeddings_test_clustering,
    )

    need_base_model = (
        online_training
        or need_recompute_embeddings_train
        or need_recompute_embeddings_train_clustering
        or need_recompute_embeddings_test_clustering
    )
    logger.info(f"Need to load base model: {need_base_model}")

    need_raw_dataloaders = (
        online_training
        or need_recompute_embeddings_train
        or need_recompute_embeddings_train_clustering
        or need_recompute_embeddings_test_clustering
    )
    logger.info(f"Need to build raw dataloaders: {need_raw_dataloaders}")

    # ------------------------------------------------------------------ #
    #  Dataloaders (raw audio)
    # ------------------------------------------------------------------ #
    # Ensure dataset_audio_max_length is defined regardless of loader needs
    dataset_audio_max_length = getattr(dataset_cfg, "audio_max_length_seconds", None)
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
        # dataset_audio_max_length already obtained above
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
            raise ValueError("Could not determine number of labels from embeddings or raw dataloaders")

        # Check if we can reuse cached model
        experiment_cfg.checkpoint_path = experiment_cfg.checkpoint_path

        # Check if we can reuse cached model (only for frozen models)
        if (
            freeze_backbone
            and cached_model is not None
            and cached_model_metadata is not None
            and cached_model_metadata.get("checkpoint_path") == experiment_cfg.checkpoint_path
            and cached_model_metadata.get("freeze_backbone") == str(freeze_backbone)
        ):
            logger.info("Reusing cached model from previous dataset (freeze_backbone=True)")
            base_model = cached_model
        else:
            logger.info("Loading model (cache miss or first dataset)")
            # Build backbone-only model; classifier heads are handled by probes
            base_model = build_model_from_spec(
                run_cfg.model_spec,
                device=str(device),
            ).to(device)

            if experiment_cfg.checkpoint_path:
                ckpt_path = anypath(experiment_cfg.checkpoint_path)
                if not exists(ckpt_path):
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

                # fs = filesystem_from_path(ckpt_path)
                # with fs.open(ckpt_path, "rb") as f:
                state = universal_torch_load(ckpt_path, map_location=device)
                # state = torch.load(f, map_location=device)

                if "model_state_dict" in state:
                    state = _process_state_dict(state)

                base_model.load_state_dict(state, strict=False)
                logger.info("Loaded checkpoint from %s", ckpt_path)

        # Note: Base model parameter freezing/counting handled by
        # build_probe_from_config() function
        # when creating the probe model, so we don't need to set it here

        # Update cached model metadata for next iteration
        if freeze_backbone:
            cached_model_metadata = {
                "checkpoint_path": experiment_cfg.checkpoint_path,
                "freeze_backbone": str(freeze_backbone),
            }
        else:
            # Don't cache fine-tuning models since weights get modified
            cached_model_metadata = None

    # ------------------------------------------------------------------ #
    #  Layer selection
    # ------------------------------------------------------------------ #
    # Do not reset layer_names when base_model is None; keep experiment config
    if base_model is not None:
        layer_names = experiment_cfg.get_target_layers()
    logger.info(f"Target layers for experiment: {layer_names}")

    # ------------------------------------------------------------------ #
    #  Calculate target_length for probes
    # ------------------------------------------------------------------ #
    target_length = None
    if dataset_audio_max_length is not None:
        # Convert audio_max_length_seconds to samples using the model's sample rate
        target_length = int(dataset_audio_max_length * run_cfg.model_spec.audio_config.sample_rate)
        logger.info(
            f"Using dataset-specific target_length: {target_length} samples "
            f"({dataset_audio_max_length}s * "
            f"{run_cfg.model_spec.audio_config.sample_rate} Hz)"
        )
    else:
        logger.info("No dataset audio_max_length_seconds specified, using default target_length")

    # ------------------------------------------------------------------ #
    #  Determine disable_layerdrop parameter for BEATs models
    # ------------------------------------------------------------------ #
    # For BEATs models, disable layerdrop to ensure consistent behavior and avoid
    # the layerdrop issue that causes hook failures
    disable_layerdrop_for_embeddings = None
    if base_model is not None and hasattr(base_model, "disable_layerdrop"):
        # For BEATs models, set disable_layerdrop=True to prevent layerdrop issues
        base_model.disable_layerdrop = True
        disable_layerdrop_for_embeddings = True
        logger.info("Setting disable_layerdrop=True for BEATs model to prevent layerdrop issues")

    test_embeds: torch.Tensor | None = None
    test_labels: torch.Tensor | None = None
    train_embeds: torch.Tensor | None = None
    train_labels: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    #  (1) Probing
    # ------------------------------------------------------------------ #

    # ------------------- embeddings for probing ------------------- #
    if not online_training:
        # Configure unified data sources (streaming vs in-memory decided internally)
        memory_limit_gb = getattr(eval_cfg.offline_embeddings, "memory_limit_gb", 32)
        memory_limit_bytes = int(memory_limit_gb * 1024**3)

        base_cfg_common = dict(
            memory_limit_bytes=memory_limit_bytes,
            use_streaming_embeddings=eval_cfg.offline_embeddings.use_streaming_embeddings,
            cache_size_limit_gb=getattr(eval_cfg.offline_embeddings, "cache_size_limit_gb", 8.0),
            chunk_size=eval_cfg.offline_embeddings.streaming_chunk_size,
            compression=eval_cfg.offline_embeddings.hdf5_compression,
            compression_level=eval_cfg.offline_embeddings.hdf5_compression_level,
            auto_chunk_size=eval_cfg.offline_embeddings.auto_chunk_size,
            max_chunk_size=eval_cfg.offline_embeddings.max_chunk_size,
            min_chunk_size=eval_cfg.offline_embeddings.min_chunk_size,
            batch_chunk_size=eval_cfg.offline_embeddings.batch_chunk_size,
            disable_tqdm=eval_cfg.disable_tqdm,
            disable_layerdrop=disable_layerdrop_for_embeddings,
        )

        train_src = EmbeddingDataSource(
            save_path=train_path,
            layer_names=layer_names,
            aggregation=aggregation_method,
            config=EmbeddingDataSourceConfig(save_path=train_path, **base_cfg_common),
        )
        val_src = EmbeddingDataSource(
            save_path=val_path,
            layer_names=layer_names,
            aggregation=aggregation_method,
            config=EmbeddingDataSourceConfig(save_path=val_path, **base_cfg_common),
        )
        test_src = EmbeddingDataSource(
            save_path=test_path,
            layer_names=layer_names,
            aggregation=aggregation_method,
            config=EmbeddingDataSourceConfig(save_path=test_path, **base_cfg_common),
        )

        # Only run probing section if we actually need probing
        if need_probe:
            if need_recompute_embeddings_train:
                train_ds = train_src.get_dataset(base_model=base_model, dataloader=train_dl_raw, device=device)
                val_ds = val_src.get_dataset(base_model=base_model, dataloader=val_dl_raw, device=device)
                test_ds = test_src.get_dataset(base_model=base_model, dataloader=test_dl_raw, device=device)
            else:
                # Fallback: try to load from existing files
                train_ds = train_src.get_dataset()
                val_ds = val_src.get_dataset()
                test_ds = test_src.get_dataset()
        else:
            # When not doing probing, set datasets to None
            train_ds = None
            val_ds = None
            test_ds = None

        if num_labels is None and need_probe:
            # Prefer num_labels computed by the data source
            num_labels = getattr(train_src, "num_labels", None)
            if num_labels is None:
                raise ValueError(
                    "num_labels could not be determined from EmbeddingDataSource; "
                    "ensure embeddings were saved with num_labels or provide it "
                    "explicitly."
                )

    # ------------------------------------------------------------------ #
    #  Experiment logger
    # ------------------------------------------------------------------ #
    exp_logger = ExperimentLogger.from_config(experiment_cfg)
    log_dir_name = evaluation_dataset_name or dataset_name
    exp_logger.log_dir = save_dir / experiment_name / log_dir_name
    exp_logger.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Probing training and evaluation
    # ------------------------------------------------------------------ #
    train_metrics: Dict[str, float] = {}
    val_metrics: Dict[str, float] = {}
    probe_test_metrics: Dict[str, float] = {}

    if "probe" in eval_cfg.eval_modes:
        dataset_metrics = evaluation_set_metrics or getattr(dataset_cfg, "metrics", None)

        # TODO: metrics per task-group
        classification_metrics = [m for m in dataset_metrics if not m.startswith("clustering_")]

        if not online_training and need_probe:
            logger.info("Training offline")
            # Get multi-label setting from dataset config, with fallback based on type
            is_multi_label = getattr(
                dataset_cfg,
                "multi_label",
                getattr(dataset_cfg, "type", None) == "detection",
            )
            # Provide embedding dims for offline training from data source
            train_embeds_dims_for_offline = getattr(train_src, "embedding_dims", None)
            (
                train_metrics,
                val_metrics,
                probe_test_metrics,
            ) = train_and_eval_offline(
                train_ds,
                val_ds,
                test_ds,
                train_embeds_dims_for_offline,
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

            # Print learned weights if the probe supports it
            if hasattr(exp_logger, "probe_model") and hasattr(exp_logger.probe_model, "get_learned_weights_table"):
                logger.info("Printing learned weights for probe:")
                weights_table = exp_logger.probe_model.get_learned_weights_table()
                logger.info(weights_table)
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

            # Log total trainable parameters for online training
            logger.info("Online training completed - parameters logged during creation")

            # Print learned weights if the probe supports it
            if hasattr(exp_logger, "probe_model") and hasattr(exp_logger.probe_model, "get_learned_weights_table"):
                logger.info("Printing learned weights for probe:")
                weights_table = exp_logger.probe_model.get_learned_weights_table()
                logger.info(weights_table)
    else:
        logger.info("Probe not run because not in eval_modes")

    _training_duration = time.time() - _training_start_time
    logger.info(
        "Training completed in %.2fs [dataset=%s, eval_set=%s, exp=%s]",
        _training_duration,
        dataset_name,
        evaluation_dataset_name,
        experiment_name,
    )
    # Log as a final train metric for easy aggregation
    exp_logger.log_metrics(
        {"training_total_duration_s": _training_duration},
        step=0,
        split="train_final",
    )

    # ------------------------------------------------------------------ #
    #  (2) Retrieval and Clustering
    # ------------------------------------------------------------------ #

    # ------------------- embeddings for train-vs-test retrieval -------- #
    if need_retrieval and retrieval_mode == "train_vs_test":
        if not need_recompute_embeddings_train_clustering:
            train_embeds_dict, train_labels, _ = load_embeddings_arrays(train_path_clustering)

            # Extract the last layer for evaluation (most processed features)
            if isinstance(train_embeds_dict, dict):
                last_layer_name = list(train_embeds_dict.keys())[-1]
                train_embeds = train_embeds_dict[last_layer_name]
                logger.info(f"Using layer '{last_layer_name}' for retrieval evaluation")
            else:
                train_embeds = train_embeds_dict
        else:
            if base_model is None:
                raise ValueError("base_model is required to compute embeddings")

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

            logger.info(f"Using EmbeddingDataSource for train embeddings (retrieval) (layers: {len(layer_names)})")
            # Use in-memory configuration for clustering/retrieval
            retrieval_cfg_common = dict(
                memory_limit_bytes=memory_limit_bytes,
                use_streaming_embeddings=False,  # Always use in-memory for retrieval
                cache_size_limit_gb=getattr(eval_cfg.offline_embeddings, "cache_size_limit_gb", 8.0),
                chunk_size=eval_cfg.offline_embeddings.streaming_chunk_size,
                compression=eval_cfg.offline_embeddings.hdf5_compression,
                compression_level=eval_cfg.offline_embeddings.hdf5_compression_level,
                auto_chunk_size=eval_cfg.offline_embeddings.auto_chunk_size,
                max_chunk_size=eval_cfg.offline_embeddings.max_chunk_size,
                min_chunk_size=eval_cfg.offline_embeddings.min_chunk_size,
                batch_chunk_size=eval_cfg.offline_embeddings.batch_chunk_size,
                disable_tqdm=eval_cfg.disable_tqdm,
                disable_layerdrop=disable_layerdrop_for_embeddings,
            )
            # Use EmbeddingDataSource for consistency with training phase
            train_src_retrieval = EmbeddingDataSource(
                save_path=train_path_clustering,
                layer_names=layer_names,
                aggregation=aggregation_method_retrieval,
                config=EmbeddingDataSourceConfig(save_path=train_path_clustering, **retrieval_cfg_common),
            )
            train_ds_retrieval = train_src_retrieval.get_dataset(
                base_model=base_model,
                dataloader=train_dl_raw,
                device=device,
            )

            # EmbeddingDataSource already saved the embeddings, just extract
            # for retrieval
            # Get the first sample to determine the structure
            sample = train_ds_retrieval[0]
            if isinstance(sample, dict):
                # Multi-layer case - use the first layer
                first_layer_name = list(sample.keys())[0] if sample else "embed"
                train_embeds = torch.stack(
                    [train_ds_retrieval[i][first_layer_name] for i in range(len(train_ds_retrieval))]
                )
                logger.info(f"Using layer '{first_layer_name}' for retrieval evaluation")
            else:
                # Single tensor case
                train_embeds = torch.stack([train_ds_retrieval[i] for i in range(len(train_ds_retrieval))])

    # ------------------- embeddings for retrieval and clustering -------- #
    test_labels: Optional[torch.Tensor] = None
    if need_retrieval or need_clustering:
        # Use the regular test path - filename encoding handles different
        # aggregation methods
        test_embeds_path = test_path_clustering
        logger.info(f"Test embeddings path: {test_embeds_path}")

        if (not overwrite) and test_embeds_path.exists():
            test_embeds_dict, test_labels, _ = load_embeddings_arrays(test_embeds_path)

            # Extract the last layer for evaluation (most processed features)
            if isinstance(test_embeds_dict, dict):
                last_layer_name = list(test_embeds_dict.keys())[-1]
                test_embeds = test_embeds_dict[last_layer_name]
                logger.info(f"Using layer '{last_layer_name}' for test evaluation")
            else:
                test_embeds = test_embeds_dict
        else:
            if base_model is None:
                raise ValueError("base_model is required to compute embeddings")

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

            logger.info(f"Using EmbeddingDataSource for test embeddings (retrieval) (layers: {len(layer_names)})")
            # Use in-memory configuration for clustering/retrieval
            retrieval_cfg_common = dict(
                memory_limit_bytes=memory_limit_bytes,
                use_streaming_embeddings=False,  # Always use in-memory for retrieval
                cache_size_limit_gb=getattr(eval_cfg.offline_embeddings, "cache_size_limit_gb", 8.0),
                chunk_size=eval_cfg.offline_embeddings.streaming_chunk_size,
                compression=eval_cfg.offline_embeddings.hdf5_compression,
                compression_level=eval_cfg.offline_embeddings.hdf5_compression_level,
                auto_chunk_size=eval_cfg.offline_embeddings.auto_chunk_size,
                max_chunk_size=eval_cfg.offline_embeddings.max_chunk_size,
                min_chunk_size=eval_cfg.offline_embeddings.min_chunk_size,
                batch_chunk_size=eval_cfg.offline_embeddings.batch_chunk_size,
                disable_tqdm=eval_cfg.disable_tqdm,
                disable_layerdrop=disable_layerdrop_for_embeddings,
            )
            # Use EmbeddingDataSource for consistency with training phase
            test_src_retrieval = EmbeddingDataSource(
                save_path=test_embeds_path,
                layer_names=layer_names,
                aggregation=aggregation_method_retrieval,
                config=EmbeddingDataSourceConfig(save_path=test_embeds_path, **retrieval_cfg_common),
            )
            test_ds_retrieval = test_src_retrieval.get_dataset(
                base_model=base_model,
                dataloader=test_dl_raw,
                device=device,
            )

            # EmbeddingDataSource already saved the embeddings, just extract
            # for retrieval
            # Get the first sample to determine the structure
            sample = test_ds_retrieval[0]
            if isinstance(sample, dict):
                # Multi-layer case - use the last layer
                last_layer_name = list(sample.keys())[-1] if sample else "embed"
                # Exclude 'label' key when extracting layer names
                layer_keys = [k for k in sample.keys() if k != "label"]
                if last_layer_name == "label":
                    last_layer_name = layer_keys[-1] if layer_keys else "embed"
                test_embeds = torch.stack(
                    [test_ds_retrieval[i][last_layer_name] for i in range(len(test_ds_retrieval))]
                )
                # Extract labels from dataset
                test_labels = torch.stack([test_ds_retrieval[i]["label"] for i in range(len(test_ds_retrieval))])
                logger.info(f"Using layer '{last_layer_name}' for test evaluation")
            else:
                # Single tensor case (shouldn't happen with EmbeddingDataset, but handle it)
                test_embeds = torch.stack([test_ds_retrieval[i] for i in range(len(test_ds_retrieval))])
                # Try to extract labels if available
                if hasattr(test_ds_retrieval, "labels"):
                    test_labels = test_ds_retrieval.labels
                else:
                    test_labels = None

        num_labels = len(test_labels.unique()) if num_labels is None else num_labels

    # ------------------------------------------------------------------ #
    #  Retrieval (from cached test embeddings)
    # ------------------------------------------------------------------ #
    retrieval_metrics: Dict[str, float] = {}
    if "retrieval" in eval_cfg.eval_modes:
        if retrieval_mode == "train_vs_test":
            if train_embeds is None:
                raise ValueError("train_embeds is required for train_vs_test retrieval")
            retrieval_metrics = eval_retrieval_cross_set(train_embeds, train_labels, test_embeds, test_labels)
        else:
            retrieval_metrics = eval_retrieval(test_embeds, test_labels)

    # ------------------------------------------------------------------ #
    #  Clustering (from cached test embeddings)
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
        checkpoint_name=(Path(experiment_cfg.checkpoint_path).name if experiment_cfg.checkpoint_path else "None"),
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
        model=base_model if freeze_backbone else None,  # Only cache frozen models
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
        logger.error("No evaluation sets found in BenchmarkEvaluationConfig. Nothing to evaluate.")
        return
    logger.info(f"Loaded {len(evaluation_sets)} evaluation sets")

    # 2. Output dir & device
    if isinstance(str(eval_cfg.save_dir), PureCloudPath):
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
        logger.warning("No evaluation sets found in BenchmarkEvaluationConfig. Nothing to evaluate.")
        return

    # Group by experiment to load each model only once (saved a lot of time.)
    for exp_cfg in eval_cfg.experiments:
        logger.info(f"Starting experiment: {exp_cfg.run_name}")

        # Log experiment probe configuration
        if exp_cfg.probe_config:
            logger.info(
                f"Experiment '{exp_cfg.run_name}' probe config: "
                f"type={exp_cfg.probe_config.probe_type}, "
                f"layers={exp_cfg.probe_config.target_layers}, "
                f"aggregation={exp_cfg.probe_config.aggregation}, "
                f"input_processing={exp_cfg.probe_config.input_processing}, "
                f"freeze_backbone={exp_cfg.probe_config.freeze_backbone}, "
                f"online_training={exp_cfg.probe_config.online_training}"
            )
        else:
            logger.info(f"Experiment '{exp_cfg.run_name}' using legacy probe configuration")

        # Log training parameters
        training_params = eval_cfg.training_params
        logger.info(
            f"Experiment '{exp_cfg.run_name}' training parameters: "
            f"epochs={training_params.train_epochs}, "
            f"lr={training_params.lr}, "
            f"batch_size={training_params.batch_size}, "
            f"optimizer={training_params.optimizer}, "
            f"weight_decay={training_params.weight_decay}, "
            f"amp={training_params.amp}, "
            f"amp_dtype={training_params.amp_dtype}, "
            f"gradient_clip_val={training_params.gradient_clip_val}, "
            f"warmup_epochs={training_params.warmup_epochs}, "
            f"scheduler_type={training_params.scheduler_type}"
        )

        cached_model = None
        model_metadata = None

        for eval_set_name, _eval_set_data_cfg in evaluation_sets:
            logger.info(f"Evaluating experiment '{exp_cfg.run_name}' on set: {eval_set_name}")

            # Extract the test dataset from the evaluation set
            test_datasets = _eval_set_data_cfg.test_datasets or []
            if not test_datasets:
                logger.warning(f"No test datasets in evaluation set '{eval_set_name}'. Skipping.")
                continue

            # For benchmark evaluation sets, we expect exactly one test dataset per set
            test_ds_cfg = test_datasets[0]

            # Get metrics from the benchmark evaluation config
            eval_set_metrics = benchmark_eval_cfg.get_metrics_for_evaluation_set(eval_set_name)

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
