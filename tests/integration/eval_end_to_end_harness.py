"""Shared helpers for `run_evaluate` end-to-end integration tests.

This module is imported by `test_run_evaluate_end_to_end.py` and by
`scripts/record_evaluate_end_to_end_metrics.py` so the same minimal probe
pipeline runs in tests and when recording per-Python metric baselines.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

import pandas as pd
import yaml

EVAL_SUMMARY_METRIC_KEYS: Final[tuple[str, ...]] = (
    "test_accuracy",
    "test_balanced_accuracy",
)


def project_root() -> Path:
    """Return repository root (parent of `avex/`).

    Returns:
        Repository root directory.
    """
    return Path(__file__).resolve().parent.parent.parent


def python_metrics_profile() -> str:
    """Return metric-baseline profile for the running interpreter.

    Returns:
        ``py310_312`` for Python 3.10–3.12, ``py313_plus`` for 3.13 and newer.
        Kept aligned with output fingerprint profiles in
        ``test_official_models_output_regression.py``.
    """
    vi = sys.version_info
    if vi < (3, 13):
        return "py310_312"
    return "py313_plus"


def create_test_data_config(data_config_path: Path) -> None:
    """Write a tiny beans-based dataset YAML for fast evaluation tests.

    Args:
        data_config_path: Destination path for the dataset config file.
    """
    test_data_config = {
        "benchmark_name": "bioacoustic_benchmark_single_test",
        "evaluation_sets": [
            {
                "name": "tiny_test",
                "train": {
                    "dataset_name": "beans",
                    "split": "dogs_train",
                    "type": "classification",
                    "label_column": "label",
                    "audio_path_col": "path",
                    "multi_label": False,
                    "label_type": "supervised",
                    "audio_max_length_seconds": 1,
                    "transformations": [
                        {"type": "rl_subsample", "ratio": 0.05, "max_samples": 10},
                        {
                            "type": "label_from_feature",
                            "feature": "label",
                            "override": True,
                        },
                    ],
                },
                "validation": {
                    "dataset_name": "beans",
                    "split": "dogs_validation",
                    "type": "classification",
                    "label_column": "label",
                    "audio_path_col": "path",
                    "multi_label": False,
                    "label_type": "supervised",
                    "audio_max_length_seconds": 1,
                    "transformations": [
                        {"type": "rl_subsample", "ratio": 0.05, "max_samples": 8},
                        {
                            "type": "label_from_feature",
                            "feature": "label",
                            "override": True,
                        },
                    ],
                },
                "test": {
                    "dataset_name": "beans",
                    "split": "dogs_test",
                    "type": "classification",
                    "label_column": "label",
                    "audio_path_col": "path",
                    "multi_label": False,
                    "label_type": "supervised",
                    "audio_max_length_seconds": 1,
                    "transformations": [
                        {"type": "rl_subsample", "ratio": 0.05, "max_samples": 8},
                        {
                            "type": "label_from_feature",
                            "feature": "label",
                            "override": True,
                        },
                    ],
                },
                "metrics": ["accuracy", "balanced_accuracy"],
            }
        ],
    }
    with open(data_config_path, "w", encoding="utf-8") as f:
        yaml.dump(test_data_config, f, default_flow_style=False)


def create_probe_eval_config(
    temp_output_dir: Path,
    probe_type: str,
    freeze_backbone: bool,
    layers: str,
    training_mode: str,
) -> Path:
    """Write an evaluation YAML for the given probe experiment settings.

    Args:
        temp_output_dir: Scratch directory for configs and default save tree.
        probe_type: e.g. ``linear`` or ``attention``.
        freeze_backbone: Whether the backbone stays frozen in the experiment.
        layers: ``last_layer`` (only variant used by the minimal test).
        training_mode: ``offline`` or ``online``.

    Returns:
        Path to the generated evaluation config YAML.
    """
    config_path = temp_output_dir / f"test_config_{probe_type}_{freeze_backbone}_{layers}_{training_mode}.yml"
    data_config_path = temp_output_dir / "test_data_config.yml"
    create_test_data_config(data_config_path)

    base_config: dict[str, object] = {
        "dataset_config": str(data_config_path),
        "training_params": {
            "train_epochs": 1,
            "lr": 0.0003,
            "batch_size": 2,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "amp": False,
            "amp_dtype": "bf16",
        },
        "save_dir": str(temp_output_dir / "results"),
        "device": "cpu",
        "seed": 42,
        "num_workers": 0,
        "eval_modes": ["probe"],
        "offline_embeddings": {
            "overwrite_embeddings": True,
            "use_streaming_embeddings": False,
            "memory_limit_gb": 2,
            "streaming_chunk_size": 100,
            "hdf5_compression": "gzip",
            "hdf5_compression_level": 4,
            "auto_chunk_size": True,
            "max_chunk_size": 200,
            "min_chunk_size": 100,
            "batch_chunk_size": 5,
            "cache_size_limit_gb": 1,
        },
    }

    target_layers = ["last_layer"] if layers == "last_layer" else ["last_layer"]
    root = project_root()
    run_config_path = (root / "configs/run_configs/aaai_train/sl_efficientnet_animalspeak.yml").resolve()

    probe_cfg: dict[str, object] = {
        "probe_type": probe_type,
        "aggregation": "mean",
        "input_processing": "pooled",
        "target_layers": target_layers,
        "freeze_backbone": freeze_backbone,
    }
    if probe_type == "linear":
        probe_cfg.update({"hidden_dims": [32], "dropout_rate": 0.1, "activation": "relu"})
    elif probe_type == "attention":
        probe_cfg.update(
            {
                "num_heads": 4,
                "attention_dim": 256,
                "num_layers": 2,
                "dropout_rate": 0.1,
                "activation": "relu",
            }
        )
    if training_mode == "offline":
        probe_cfg["freeze_backbone"] = True
    else:
        probe_cfg["freeze_backbone"] = False

    experiment: dict[str, object] = {
        "run_name": f"{probe_type}_{freeze_backbone}_{layers}_{training_mode}",
        "run_config": str(run_config_path),
        "probe_config": probe_cfg,
        "pretrained": True,
    }

    base_config["experiments"] = [experiment]

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(base_config, f, default_flow_style=False)

    return config_path


def run_linear_offline_probe_evaluate(temp_output_dir: Path) -> dict[str, float]:
    """Run the minimal linear / offline / last_layer evaluate job and read metrics.

    Mirrors ``test_weighted_probes_comprehensive`` (same patches and paths).

    Args:
        temp_output_dir: Temporary directory for configs and run outputs.

    Returns:
        Mapping of summary CSV column names to scalar metric values.
    """
    from avex.run_evaluate import main

    probe_type, freeze_backbone, layers, training_mode = ("linear", True, "last_layer", "offline")
    config_path = create_probe_eval_config(temp_output_dir, probe_type, freeze_backbone, layers, training_mode)
    test_output_dir = temp_output_dir / f"{probe_type}_{freeze_backbone}_{layers}_{training_mode}"
    test_output_dir.mkdir(exist_ok=True)

    patches = (
        f"save_dir={test_output_dir}",
        "device=cpu",
        "seed=42",
        "training_params.train_epochs=1",
        "training_params.batch_size=1",
        "offline_embeddings.use_streaming_embeddings=false",
        "offline_embeddings.memory_limit_gb=2",
        "offline_embeddings.cache_size_limit_gb=1",
    )

    root = project_root()
    original_cwd = os.getcwd()
    try:
        os.chdir(root)
        main(config_path, patches)
    finally:
        os.chdir(original_cwd)

    summary_csvs = list(test_output_dir.rglob("*summary*.csv"))
    assert summary_csvs, f"No summary CSVs found in {test_output_dir}"
    summary_csv = summary_csvs[0]
    df = pd.read_csv(summary_csv)

    for metric in EVAL_SUMMARY_METRIC_KEYS:
        assert metric in df.columns, f"Missing metric column: {metric}"

    out: dict[str, float] = {}
    for metric in EVAL_SUMMARY_METRIC_KEYS:
        val = df[metric].iloc[0]
        if pd.isna(val):
            continue
        out[metric] = float(val)
    return out
