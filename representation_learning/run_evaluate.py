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

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from esp_data import DatasetConfig
from esp_data.io import anypath

from representation_learning.configs import (
    EvaluateConfig,
    ExperimentConfig,
    RunConfig,
    load_config,
)
from representation_learning.data.dataset import build_dataloaders
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
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    probe_test_metrics: Dict[str, float]
    retrieval_metrics: Dict[str, float]


# -------------------------------------------------------------------- #
#  CLI
# -------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Linear-probe / fine-tune an audio model")
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to evaluation YAML (see configs/evaluation_configs/*)",
    )
    return p.parse_args()


# -------------------------------------------------------------------- #
#  Core routine for one (dataset, experiment) pair
# -------------------------------------------------------------------- #
def run_experiment(
    eval_cfg: EvaluateConfig,
    dataset_cfg: DatasetConfig,
    experiment_cfg: ExperimentConfig,
    device: torch.device,
    save_dir: Path,
) -> ExperimentResult:
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
    run_cfg.model_spec.device = device
    run_cfg.augmentations = []  # disable training-time noise / mix-up

    dataset_cfg.sample_rate = run_cfg.model_spec.audio_config.sample_rate

    # Embedding paths
    emb_base_dir = save_dir / experiment_name / dataset_name
    train_path = emb_base_dir / "embedding_train.h5"
    val_path = emb_base_dir / "embedding_val.h5"
    test_path = emb_base_dir / "embedding_test.h5"

    # Flags to determine which embeddings we need
    need_probe = "linear_probe" in eval_cfg.eval_modes
    need_retrieval = "retrieval" in eval_cfg.eval_modes

    overwrite = getattr(eval_cfg, "overwrite_embeddings", False)
    need_recompute_embeddings_train = overwrite or (
        need_probe and not (train_path.exists() and val_path.exists())
    )
    need_recompute_embeddings_test = overwrite or (
        (need_retrieval or need_probe) and not test_path.exists()
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
        train_dl_raw, val_dl_raw, test_dl_raw = build_dataloaders(
            run_cfg, dataset_cfg, device
        )
        logger.info(
            "Raw dataloaders ready: %d/%d/%d raw batches",
            len(train_dl_raw),
            len(val_dl_raw),
            len(test_dl_raw),
        )
        num_labels = len(train_dl_raw.dataset.metadata["label_map"])

    # ------------------------------------------------------------------ #
    #  Backbone (optionally load checkpoint)
    # ------------------------------------------------------------------ #
    base_model: Optional[torch.nn.Module] = None

    if need_base_model:
        if num_labels is None:
            # Try to get num_labels from existing embeddings
            if train_path.exists():
                _, _, num_labels = load_embeddings_arrays(train_path)
            elif test_path.exists():
                _, _, num_labels = load_embeddings_arrays(test_path)
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
                base_model.load_state_dict(state["model_state_dict"], strict=False)
            else:
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

    # ------------------- embeddings for retrieval ---------------------- #
    if need_retrieval or (need_probe and frozen):
        if not need_recompute_embeddings_test:
            test_embeds, test_labels, num_labels = load_embeddings_arrays(test_path)
        else:
            if base_model is None:
                raise ValueError("base_model is required to compute embeddings")
            test_embeds, test_labels = extract_embeddings_for_split(
                base_model, test_dl_raw, layer_names, device
            )
            save_embeddings_arrays(test_embeds, test_labels, test_path, num_labels)

    # ------------------------------------------------------------------ #
    #  Experiment logger
    # ------------------------------------------------------------------ #
    exp_logger = ExperimentLogger.from_config(experiment_cfg)
    exp_logger.log_dir = save_dir / experiment_name / dataset_name
    exp_logger.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  (1) Linear probe
    # ------------------------------------------------------------------ #
    train_metrics: Dict[str, float] = {}
    val_metrics: Dict[str, float] = {}
    probe_test_metrics: Dict[str, float] = {}

    if "linear_probe" in eval_cfg.eval_modes:
        if frozen:
            (
                train_metrics,
                val_metrics,
                probe_test_metrics,
            ) = train_and_eval_linear_probe(
                train_ds,
                val_ds,
                EmbeddingDataset(test_embeds, test_labels),
                base_model,
                num_labels,
                layer_names,
                eval_cfg,
                device,
                exp_logger,
                dataset_cfg.multi_label,
            )
        else:
            # For fine-tuning, use raw dataloaders
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
                dataset_cfg.multi_label,
            )
    else:
        logger.info("Linear probe not run because not in eval_modes")

    # ------------------------------------------------------------------ #
    #  (2) Retrieval (from cached test embeddings)
    # ------------------------------------------------------------------ #
    retrieval_metrics: Dict[str, float] = {}
    if "retrieval" in eval_cfg.eval_modes:
        retrieval_metrics = eval_retrieval(test_embeds, test_labels)

    # Log & finish
    exp_logger.log_metrics(train_metrics, step=0, split="train_final")
    exp_logger.log_metrics(val_metrics, step=0, split="val_final")
    exp_logger.log_metrics(probe_test_metrics, step=0, split="test_probe")
    exp_logger.log_metrics(retrieval_metrics, step=0, split="test_retrieval")
    exp_logger.finalize()

    return ExperimentResult(
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        probe_test_metrics=probe_test_metrics,
        retrieval_metrics=retrieval_metrics,
    )


# -------------------------------------------------------------------- #
#  Main
# -------------------------------------------------------------------- #
def main() -> None:
    args = _parse_args()

    # 1. Load configs
    eval_cfg: EvaluateConfig = load_config(args.config, config_type="evaluate")
    benchmark_cfg = load_config(eval_cfg.dataset_config, config_type="benchmark")

    # 2. Output dir & device
    if str(eval_cfg.save_dir).startswith("gs://"):
        save_dir = anypath(str(eval_cfg.save_dir))
        # For GCS paths we rely on cloudpathlib to create objects lazily when
        # data is written, so no mkdir is needed here.
    else:
        save_dir = Path(str(eval_cfg.save_dir)).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 3. Run all (dataset × experiment)
    all_results: List[ExperimentResult] = []
    for ds_cfg in benchmark_cfg.datasets:
        for exp_cfg in eval_cfg.experiments:
            res = run_experiment(eval_cfg, ds_cfg, exp_cfg, device, save_dir)
            all_results.append(res)

            logger.info(
                "[%s | %s]  probe-test: %s | retrieval: %s",
                res.dataset_name,
                res.experiment_name,
                res.probe_test_metrics,
                res.retrieval_metrics or "n/a",
            )

    # 4. Write summary
    summary_path = save_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write("Experiment Summary\n==================\n\n")
        for r in all_results:
            f.write(f"Dataset: {r.dataset_name}\n")
            f.write(f"Experiment: {r.experiment_name}\n")
            f.write(f"Train metrics: {r.train_metrics}\n")
            f.write(f"Validation metrics: {r.val_metrics}\n")
            f.write(f"Probe test metrics: {r.probe_test_metrics}\n")
            f.write(f"Retrieval metrics: {r.retrieval_metrics}\n")
            f.write("-" * 60 + "\n")

    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
