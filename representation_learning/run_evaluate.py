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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from cloudpathlib import GSPath

from esp_data_temp.config import DatasetConfig
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
from representation_learning.evaluation.retrieval import (
    evaluate_auc_roc,
    evaluate_precision,
)
from representation_learning.metrics.metric_factory import get_metric_class
from representation_learning.models.get_model import get_model
from representation_learning.models.linear_probe import LinearProbe
from representation_learning.training.optimisers import get_optimizer
from representation_learning.training.train import FineTuneTrainer
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
#  Linear-probe helper
# -------------------------------------------------------------------- #
def _train_and_eval_linear_probe(
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    test_embed_ds: torch.utils.data.Dataset,
    base_model: torch.nn.Module,
    num_labels: int,
    layer_names: List[str],
    eval_cfg: EvaluateConfig,
    device: torch.device,
    exp_logger: ExperimentLogger,
    multi_label: bool,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Train a linear probe and evaluate it on *cached* test embeddings.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
        • **train_metrics** – aggregated over training split
        • **val_metrics**   – aggregated over validation split
        • **probe_test_metrics** – metrics on cached-embedding test split
    """
    probe = LinearProbe(
        base_model, layer_names, num_labels, device=device, feature_mode=True
    )
    logger.info(
        "Linear probe → %d parameters", sum(p.numel() for p in probe.parameters())
    )

    if eval_cfg.frozen:
        logger.info("Freezing backbone parameters (eval_cfg.frozen=True)")
        for p in base_model.parameters():
            p.requires_grad = False
    else:
        raise NotImplementedError("Full fine-tuning not supported")

    optim = get_optimizer(probe.parameters(), eval_cfg.training_params)

    trainer = FineTuneTrainer(
        model=probe,
        optimizer=optim,
        train_loader=torch.utils.data.DataLoader(
            train_ds,
            batch_size=eval_cfg.training_params.batch_size,
            shuffle=True,
        ),
        val_loader=torch.utils.data.DataLoader(
            val_ds,
            batch_size=eval_cfg.training_params.batch_size,
            shuffle=False,
        ),
        device=device,
        cfg=eval_cfg,
        exp_logger=exp_logger,
        multi_label=multi_label,
    )

    train_metrics, val_metrics = trainer.train(
        num_epochs=eval_cfg.training_params.train_epochs
    )

    # ---------- probe evaluation on cached test embeddings ----------
    test_loader = torch.utils.data.DataLoader(
        test_embed_ds,
        batch_size=eval_cfg.training_params.batch_size,
        shuffle=False,
    )

    # Metric selection (fallback to accuracy)
    metric_names = getattr(test_embed_ds, "metadata", {}).get("metrics", ["accuracy"])
    metrics = [get_metric_class(m, num_labels) for m in metric_names]

    probe.eval()  # feature_mode stays True (inputs are embeddings)
    with torch.no_grad():
        for batch in test_loader:
            z = batch["embed"].to(device)
            y = batch["label"].to(device)

            logits = probe(z)
            for met in metrics:
                met.update(logits, y)

    probe_test_metrics = {
        name: met.get_primary_metric()
        for name, met in zip(metric_names, metrics, strict=False)
    }
    return train_metrics, val_metrics, probe_test_metrics


def _process_state_dict(state_dict: dict) -> dict:
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    # drop classifier.weight and classifier.bias safely
    state_dict.pop("classifier.weight", None)
    state_dict.pop("classifier.bias", None)
    state_dict.pop("model.classifier.1.weight", None)
    state_dict.pop("model.classifier.1.bias", None)

    return state_dict


# -------------------------------------------------------------------- #
#  Retrieval helper
# -------------------------------------------------------------------- #
def _eval_retrieval(
    embeds: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """Compute retrieval metrics using modular evaluation utilities.

    Returns
    -------
    Dict[str, float]
        ``{"retrieval_roc_auc": value, "retrieval_precision_at_1": value}``.
    """
    roc_auc = evaluate_auc_roc(embeds.numpy(), labels.numpy())
    precision_at_1 = evaluate_precision(embeds.numpy(), labels.numpy(), k=1)

    return {
        "retrieval_roc_auc": roc_auc,
        "retrieval_precision_at_1": precision_at_1,
    }


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
    logger.info("running experiment")
    dataset_name = dataset_cfg.dataset_name
    experiment_name = experiment_cfg.run_name
    logger.info(
        "Running experiment '%s' on dataset '%s'",
        experiment_name,
        dataset_name,
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

    # ------------------------------------------------------------------ #
    #  Dataloaders (raw audio)
    # ------------------------------------------------------------------ #
    train_dl_raw, val_dl_raw, test_dl_raw = build_dataloaders(
        run_cfg, dataset_cfg, device
    )
    logger.info(
        "Dataset ready: %d/%d/%d raw batches",
        len(train_dl_raw),
        len(val_dl_raw),
        len(test_dl_raw),
    )

    # ------------------------------------------------------------------ #
    #  Backbone (optionally load checkpoint)
    # ------------------------------------------------------------------ #
    num_labels = len(train_dl_raw.dataset.metadata["label_map"])
    base_model = get_model(run_cfg.model_spec, num_classes=num_labels).to(device)

    if experiment_cfg.checkpoint_path:
        ckpt_path = (
            GSPath(experiment_cfg.checkpoint_path)
            if experiment_cfg.checkpoint_path.startswith("gs://")
            else Path(experiment_cfg.checkpoint_path).expanduser()
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        with ckpt_path.open("rb") as f:
            state = torch.load(f, map_location=device)
        state = _process_state_dict(state)
        base_model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint from %s", ckpt_path)

    base_model.eval()

    # ------------------------------------------------------------------ #
    #  Layer selection
    # ------------------------------------------------------------------ #
    if experiment_cfg.layers == "last_layer":
        linear_layers = [
            n for n, m in base_model.named_modules() if isinstance(m, torch.nn.Linear)
        ]
        layer_names = [linear_layers[-1]]
    else:
        layer_names = experiment_cfg.layers.split(",")

    # Flags to determine which embeddings we need
    need_probe = "linear_probe" in eval_cfg.eval_modes
    need_retrieval = "retrieval" in eval_cfg.eval_modes

    overwrite = getattr(eval_cfg, "overwrite_embeddings", False)

    train_ds: EmbeddingDataset | None = None  # will remain None if not needed
    val_ds: EmbeddingDataset | None = None
    test_embeds: torch.Tensor | None = None
    test_labels: torch.Tensor | None = None

    emb_base_dir = save_dir / experiment_name / dataset_name

    # ------------------- embeddings for linear probe ------------------- #
    if need_probe:
        train_path = emb_base_dir / "embedding_train.h5"
        val_path = emb_base_dir / "embedding_val.h5"

        if (not overwrite) and train_path.exists() and val_path.exists():
            train_embeds, train_labels = load_embeddings_arrays(train_path)
            val_embeds, val_labels = load_embeddings_arrays(val_path)
        else:
            train_embeds, train_labels = extract_embeddings_for_split(
                base_model, train_dl_raw, layer_names, device
            )
            val_embeds, val_labels = extract_embeddings_for_split(
                base_model, val_dl_raw, layer_names, device
            )

            # Persist to disk
            save_embeddings_arrays(train_embeds, train_labels, train_path)
            save_embeddings_arrays(val_embeds, val_labels, val_path)

        train_ds = EmbeddingDataset(train_embeds, train_labels)
        val_ds = EmbeddingDataset(val_embeds, val_labels)

    # ------------------- embeddings for retrieval ---------------------- #
    if need_retrieval or need_probe:
        test_path = emb_base_dir / "embedding_test.h5"
        # test_path = Path("~/EAT/bat_embeddings_consolidated.h5").expanduser()
        # print("exists", test_path.exists())
        # test_path = Path("~/EAT/cbi_embeddings_consolidated.h5")
        logger.info(test_path)

        if (not overwrite) and test_path.exists():
            test_embeds, test_labels = load_embeddings_arrays(test_path)
            print(test_embeds.shape, test_labels[0])
        else:
            test_embeds, test_labels = extract_embeddings_for_split(
                base_model, test_dl_raw, layer_names, device
            )
            save_embeddings_arrays(test_embeds, test_labels, test_path)

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
        (
            train_metrics,
            val_metrics,
            probe_test_metrics,
        ) = _train_and_eval_linear_probe(
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

    # ------------------------------------------------------------------ #
    #  (2) Retrieval (from cached test embeddings)
    # ------------------------------------------------------------------ #
    retrieval_metrics: Dict[str, float] = {}
    if "retrieval" in eval_cfg.eval_modes:
        retrieval_metrics = _eval_retrieval(test_embeds, test_labels)
        # logger.info("retrieval metrics", retrieval_metrics)

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
    # logger.info("eval cfg", eval_cfg)

    # 2. Output dir & device
    if str(eval_cfg.save_dir).startswith("gs://"):
        save_dir = GSPath(str(eval_cfg.save_dir))
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
    summary_path = save_dir / f"summary_{datetime.now()}.txt"
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
