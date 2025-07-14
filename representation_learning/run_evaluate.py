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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from esp_data import DatasetConfig
from esp_data.io import anypath

# Import representation_learning modules
from representation_learning.configs import (
    DatasetCollectionConfig,
    EvaluateConfig,
    ExperimentConfig,
    RunConfig,
    load_config,
)
from representation_learning.data.dataset import build_dataloaders
from representation_learning.evaluation.clustering import (
    eval_clustering,
    eval_clustering_multiple_k,
)
from representation_learning.evaluation.embedding_utils import (
    EmbeddingDataset,
    extract_embeddings_for_split,
    load_embeddings_arrays,
    save_embeddings_arrays,
)
from representation_learning.evaluation.finetune import (
    train_and_eval_full_fine_tune,
    train_and_eval_linear_probe,
)
from representation_learning.evaluation.retrieval import eval_retrieval
from representation_learning.models.get_model import get_model
from representation_learning.utils import ExperimentLogger
from representation_learning.utils.experiment_tracking import (
    create_experiment_summary_csvs,
    get_or_create_experiment_metadata,
    save_evaluation_metadata,
)
from representation_learning.utils.utils import _process_state_dict

logger = logging.getLogger("run_finetune")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


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
) -> ExperimentResult:
    logger.info("running experiment")
    dataset_name = dataset_cfg.dataset_name
    experiment_name = experiment_cfg.run_name
    frozen = experiment_cfg.frozen

    logger.info(
        "Running experiment '%s' on dataset '%s' with frozen: %s",
        experiment_name,
        dataset_name,
        frozen,
    )

    # ------------------------------------------------------------------ #
    #  Build run config
    # ------------------------------------------------------------------ #
    run_cfg: RunConfig = load_config(experiment_cfg.run_config)
    run_cfg.model_spec.audio_config.window_selection = "start"
    run_cfg.training_params = eval_cfg.training_params
    run_cfg.model_spec.device = str(device)
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
    need_probe = "linear_probe" in eval_cfg.eval_modes
    need_retrieval = "retrieval" in eval_cfg.eval_modes
    need_clustering = "clustering" in eval_cfg.eval_modes

    overwrite = getattr(eval_cfg, "overwrite_embeddings", False)
    need_recompute_embeddings_train = overwrite or (
        need_probe and not (train_path.exists() and val_path.exists())
    )
    need_recompute_embeddings_test = overwrite or (
        (need_retrieval or need_probe or need_clustering) and not test_path.exists()
    )
    logger.info(
        "Need to recompute embeddings for train: %s and test: %s",
        need_recompute_embeddings_train,
        need_recompute_embeddings_test,
    )

    need_base_model = (
        need_recompute_embeddings_train
        or need_recompute_embeddings_test
        or (not frozen and "linear_probe" in eval_cfg.eval_modes)
    )
    logger.info(f"Need to load base model: {need_base_model}")

    need_raw_dataloaders = (
        need_recompute_embeddings_train
        or need_recompute_embeddings_test
        or (not frozen and "linear_probe" in eval_cfg.eval_modes)
    )
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
            or ([dataset_cfg] if "linear_probe" in eval_cfg.eval_modes else None),
            val_datasets=data_collection_cfg.val_datasets
            or ([dataset_cfg] if "linear_probe" in eval_cfg.eval_modes else None),
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

        # For BirdSet datasets, augment during eval
        is_birdset = (
            hasattr(dataset_cfg, "train")
            and hasattr(dataset_cfg.train, "dataset_name")
            and dataset_cfg.train.dataset_name.lower() == "birdset"
        )

        if is_birdset:
            logger.info(
                f"BirdSet dataset detected ({dataset_name}). "
                "Enabling augmentations for train/validation during evaluation "
                "(test always has no augmentations)."
            )

        train_dl_raw, val_dl_raw, test_dl_raw = build_dataloaders(
            run_cfg,
            eval_data_collection_cfg,
            device,
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

        base_model = get_model(run_cfg.model_spec, num_classes=num_labels).to(device)

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

    # ------------------------------------------------------------------ #
    #  Layer selection
    # ------------------------------------------------------------------ #
    if base_model is not None:
        if experiment_cfg.layers == "last_layer":
            linear_layers = [
                n
                for n, m in base_model.named_modules()
                if isinstance(m, torch.nn.Linear)
            ]
            layer_names = [linear_layers[-1]]
        else:
            layer_names = experiment_cfg.layers.split(",")
    else:
        # When base_model is None, we don't need layer names
        layer_names = []

    train_ds: EmbeddingDataset | None = None  # will remain None if not needed
    val_ds: EmbeddingDataset | None = None
    test_embeds: torch.Tensor | None = None
    test_labels: torch.Tensor | None = None

    # ------------------- embeddings for linear probe ------------------- #
    if need_probe and frozen:
        if not need_recompute_embeddings_train:
            train_embeds, train_labels, num_labels = load_embeddings_arrays(train_path)
            val_embeds, val_labels, _ = load_embeddings_arrays(val_path)
        else:
            if base_model is None:
                raise ValueError("base_model is required to compute embeddings")
            train_embeds, train_labels = extract_embeddings_for_split(
                base_model, train_dl_raw, layer_names, device
            )
            val_embeds, val_labels = extract_embeddings_for_split(
                base_model, val_dl_raw, layer_names, device
            )

            # Persist to disk
            save_embeddings_arrays(train_embeds, train_labels, train_path, num_labels)
            save_embeddings_arrays(val_embeds, val_labels, val_path, num_labels)

        train_ds = EmbeddingDataset(train_embeds, train_labels)
        val_ds = EmbeddingDataset(val_embeds, val_labels)
        num_labels = len(train_labels.unique()) if num_labels is None else num_labels

    # ------------------- embeddings for retrieval and clustering -------- #
    if need_retrieval or need_probe or need_clustering:
        test_path = emb_base_dir / "embedding_test.h5"
        logger.info(test_path)

        if (not overwrite) and test_path.exists():
            test_embeds, test_labels, _ = load_embeddings_arrays(test_path)
        else:
            if base_model is None:
                raise ValueError("base_model is required to compute embeddings")
            test_embeds, test_labels = extract_embeddings_for_split(
                base_model, test_dl_raw, layer_names, device
            )
            save_embeddings_arrays(test_embeds, test_labels, test_path, num_labels)

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

    if "linear_probe" in eval_cfg.eval_modes:
        dataset_metrics = evaluation_set_metrics or getattr(
            dataset_cfg, "metrics", None
        )

        # TODO: metrics per task-group
        classification_metrics = [
            m for m in dataset_metrics if not m.startswith("clustering_")
        ]

        if frozen:
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
            ) = train_and_eval_linear_probe(
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
            )
        else:
            # For fine-tuning, use raw dataloaders
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
            ) = train_and_eval_full_fine_tune(
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
            )
    else:
        logger.info("Linear probe not run because not in eval_modes")

    # ------------------------------------------------------------------ #
    #  (2) Retrieval (from cached test embeddings)
    # ------------------------------------------------------------------ #
    retrieval_metrics: Dict[str, float] = {}
    if "retrieval" in eval_cfg.eval_modes:
        retrieval_metrics = eval_retrieval(test_embeds, test_labels)
        # logger.info("retrieval metrics", retrieval_metrics)

    # ------------------------------------------------------------------ #
    #  (3) Clustering (from cached test embeddings)
    # ------------------------------------------------------------------ #
    clustering_metrics: Dict[str, float] = {}
    if "clustering" in eval_cfg.eval_modes:
        # Use both standard clustering with true K and multiple K search
        clustering_metrics.update(eval_clustering(test_embeds, test_labels))
        clustering_metrics.update(eval_clustering_multiple_k(test_embeds, test_labels))
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
        checkpoint_name=Path(experiment_cfg.checkpoint_path).name
        if experiment_cfg.checkpoint_path
        else "None",
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        probe_test_metrics=probe_test_metrics,
        retrieval_metrics=retrieval_metrics,
        clustering_metrics=clustering_metrics,
        eval_config=eval_cfg.model_dump(mode="json"),
        training_metadata=training_metadata,
        run_config=run_cfg.model_dump(mode="json"),  # Always include run config
    )

    return ExperimentResult(
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        evaluation_dataset_name=evaluation_dataset_name,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        probe_test_metrics=probe_test_metrics,
        retrieval_metrics=retrieval_metrics,
        clustering_metrics=clustering_metrics,
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
    eval_cfg: EvaluateConfig = load_config(args.config, config_type="evaluate")

    # Detect config format based on content structure
    config_path = Path(eval_cfg.dataset_config)

    # Load raw config to check structure
    with open(config_path, "r") as f:
        import yaml

        raw_config = yaml.safe_load(f)

    if "evaluation_sets" in raw_config:
        # BenchmarkEvaluationConfig format
        benchmark_eval_cfg = load_config(
            eval_cfg.dataset_config, config_type="benchmark_evaluation"
        )
        evaluation_sets = benchmark_eval_cfg.get_all_evaluation_sets()
        logger.info(
            f"Loading dataset config '{eval_cfg.dataset_config}' as "
            f"'benchmark_evaluation' format with {len(evaluation_sets)} evaluation sets"
        )
    else:
        # Structured dataset collection config
        load_config(eval_cfg.dataset_config, config_type="data")
        logger.info(
            f"Loading dataset config '{eval_cfg.dataset_config}' as 'data' format"
        )

    # 2. Output dir & device
    if str(eval_cfg.save_dir).startswith("gs://"):
        save_dir = anypath(str(eval_cfg.save_dir))
    else:
        save_dir = Path(str(eval_cfg.save_dir)).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 3. Run experiments
    all_results: List[ExperimentResult] = []

    if not evaluation_sets:
        logger.warning(
            "No evaluation sets found in BenchmarkEvaluationConfig. "
            "Nothing to evaluate."
        )
        return

    for eval_set_name, eval_set_data_cfg in evaluation_sets:
        logger.info(f"Evaluating benchmark set: {eval_set_name}")

        # Extract the test dataset from the evaluation set
        test_datasets = eval_set_data_cfg.test_datasets or []
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

        for exp_cfg in eval_cfg.experiments:
            res = run_experiment(
                eval_cfg,
                test_ds_cfg,
                exp_cfg,
                eval_set_data_cfg,
                device,
                save_dir,
                eval_set_name,
                eval_set_metrics,
            )
            all_results.append(res)

            logger.info(
                "[%s | %s | %s]  probe-test: %s | retrieval: %s | clustering: %s",
                eval_set_name,
                res.dataset_name,
                res.experiment_name,
                res.probe_test_metrics,
                res.retrieval_metrics or "n/a",
                res.clustering_metrics or "n/a",
            )

    # 5. Create and save summary CSV files
    create_experiment_summary_csvs(
        all_results=all_results,
        eval_cfg=eval_cfg,
        save_dir=save_dir,
        config_file_path=str(args.config),
        benchmark_eval_cfg=benchmark_eval_cfg,
        evaluation_sets=evaluation_sets,
        experiments=eval_cfg.experiments,
    )


if __name__ == "__main__":
    main()
