"""
Tests for experiment tracking utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest
from esp_data import DatasetConfig

from representation_learning.configs import (
    AudioConfig,
    DatasetCollectionConfig,
    ModelSpec,
    RunConfig,
    SchedulerConfig,
    TrainingParams,
)
from representation_learning.utils.experiment_tracking import (
    create_initial_experiment_metadata,
    get_run_config_params_from_metadata,
    get_training_params_from_metadata,
    load_evaluation_metadata,
    load_experiment_metadata,
    save_evaluation_metadata,
    save_experiment_metadata,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for testing.

    Returns:
        Path: Path to the temporary directory.
    """
    return tmp_path


@pytest.fixture
def mock_config(tmp_path: Path) -> RunConfig:
    """Create a mock RunConfig for testing.

    Returns:
        RunConfig: A mock configuration object for testing.
    """
    # Build a minimal DatasetCollectionConfig expected by RunConfig
    ds_cfg = DatasetCollectionConfig(
        val_datasets=[
            DatasetConfig(
                dataset_name="animalspeak",
                split="validation",
                audio_path_col="gs_path",
            )
        ]
    )

    return RunConfig(
        model_spec=ModelSpec(
            name="efficientnet",
            pretrained=False,
            device="cpu",
            audio_config=AudioConfig(
                sample_rate=16000,
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window="hann",
                n_mels=128,
                representation="mel_spectrogram",
                normalize=True,
                target_length_seconds=10,
                window_selection="random",
            ),
            efficientnet_variant="b0",
        ),
        training_params=TrainingParams(
            train_epochs=10,
            lr=0.001,
            batch_size=32,
            optimizer="adamw",
            weight_decay=0.01,
        ),
        dataset_config=ds_cfg,
        output_dir="test_output",
        loss_function="cross_entropy",
        scheduler=SchedulerConfig(),
        device="cpu",
        seed=42,
        num_workers=0,
        run_name="test_run",
        wandb_project="test_project",
    )


@pytest.fixture
def mock_metrics() -> Dict[str, float]:
    """Create mock metrics for testing.

    Returns:
        Dict[str, float]: Dictionary containing mock metric values.
    """
    return {
        "loss": 0.5,
        "accuracy": 0.95,
        "balanced_accuracy": 0.9,
        "retrieval_roc_auc": 0.85,
        "clustering_ari": 0.8,
    }


def test_save_and_load_experiment_metadata(
    temp_dir: Path, mock_config: RunConfig, mock_metrics: Dict[str, float]
) -> None:
    """Test saving and loading experiment metadata."""
    # Save metadata
    save_experiment_metadata(
        output_dir=temp_dir,
        config=mock_config,
        checkpoint_name="test_checkpoint.pt",
        metrics=mock_metrics,
        is_best=True,
        is_final=False,
    )
    # Load metadata
    df = load_experiment_metadata(temp_dir)

    # Verify metadata
    assert not df.empty
    assert "end_timestamp" in df.columns
    assert df["checkpoint_name"].iloc[0] == "test_checkpoint.pt"
    assert df["is_best"].iloc[0]
    assert not df["is_final"].iloc[0]
    assert df["loss"].iloc[0] == 0.5
    assert df["accuracy"].iloc[0] == 0.95

    # Verify model spec fields
    model_spec = json.loads(df["config"].iloc[0])["model_spec"]
    assert model_spec["name"] == "efficientnet"
    assert model_spec["efficientnet_variant"] == "b0"
    assert model_spec["audio_config"]["sample_rate"] == 16000


def test_save_and_load_evaluation_metadata(
    temp_dir: Path, mock_config: RunConfig, mock_metrics: Dict[str, float]
) -> None:
    """Test saving and loading evaluation metadata."""
    # Save metadata
    save_evaluation_metadata(
        output_dir=temp_dir,
        eval_config=mock_config.model_dump(mode="json"),
        checkpoint_name="test_checkpoint.pt",
        train_metrics=mock_metrics,
        val_metrics=mock_metrics,
        probe_test_metrics=mock_metrics,
        retrieval_metrics=mock_metrics,
        clustering_metrics=mock_metrics,
        dataset_name="test_dataset",
        experiment_name="test_experiment",
    )

    # Load metadata
    df = load_evaluation_metadata(temp_dir)

    # Verify metadata
    assert not df.empty
    assert "end_timestamp" in df.columns
    assert df["checkpoint_name"].iloc[0] == "test_checkpoint.pt"
    assert df["train_test_dataset_loss"].iloc[0] == 0.5
    assert df["train_test_dataset_accuracy"].iloc[0] == 0.95
    assert df["val_test_dataset_loss"].iloc[0] == 0.5
    assert df["val_test_dataset_accuracy"].iloc[0] == 0.95
    assert df["test_test_dataset_accuracy"].iloc[0] == 0.95
    assert df["test_test_dataset_retrieval_roc_auc"].iloc[0] == 0.85
    assert df["dataset_name"].iloc[0] == "test_dataset"
    assert df["experiment_name"].iloc[0] == "test_experiment"

    # Verify eval_config is stored as JSON string
    assert "eval_config" in df.columns

    eval_config = json.loads(df["eval_config"].iloc[0])
    assert eval_config["model_spec"]["name"] == "efficientnet"
    assert eval_config["model_spec"]["efficientnet_variant"] == "b0"
    assert eval_config["model_spec"]["audio_config"]["sample_rate"] == 16000


def test_create_initial_experiment_metadata(
    temp_dir: Path, mock_config: RunConfig, mock_metrics: Dict[str, float]
) -> None:
    """Test creating initial experiment metadata."""
    df = create_initial_experiment_metadata(
        output_dir=temp_dir,
        config=mock_config,
        checkpoint_name="test_checkpoint.pt",
    )

    # Verify metadata
    assert not df.empty
    assert "end_timestamp" in df.columns
    assert df["checkpoint_name"].iloc[0] == "test_checkpoint.pt"
    assert df["is_best"].iloc[0]
    assert df["is_final"].iloc[0]

    # Verify config is stored as JSON string
    assert "config" in df.columns

    config = json.loads(df["config"].iloc[0])
    assert config["model_spec"]["name"] == "efficientnet"
    assert config["model_spec"]["efficientnet_variant"] == "b0"
    assert config["model_spec"]["audio_config"]["sample_rate"] == 16000


def test_append_to_existing_metadata(
    temp_dir: Path, mock_config: RunConfig, mock_metrics: Dict[str, float]
) -> None:
    """Test appending to existing metadata."""
    # Create initial metadata
    save_experiment_metadata(
        output_dir=temp_dir,
        config=mock_config,
        checkpoint_name="checkpoint1.pt",
        metrics=mock_metrics,
        is_best=True,
        is_final=False,
    )

    # Append new metadata
    save_experiment_metadata(
        output_dir=temp_dir,
        config=mock_config,
        checkpoint_name="checkpoint2.pt",
        metrics=mock_metrics,
        is_best=False,
        is_final=True,
    )

    # Load metadata
    df = load_experiment_metadata(temp_dir)

    # Verify both entries exist
    assert len(df) == 2
    assert df["checkpoint_name"].iloc[0] == "checkpoint1.pt"
    assert df["checkpoint_name"].iloc[1] == "checkpoint2.pt"
    assert df["is_best"].iloc[0]
    assert not df["is_best"].iloc[1]
    assert not df["is_final"].iloc[0]
    assert df["is_final"].iloc[1]
    assert df["loss"].iloc[0] == 0.5
    assert df["accuracy"].iloc[0] == 0.95

    # Verify model spec fields
    model_spec = json.loads(df["config"].iloc[0])["model_spec"]
    assert model_spec["name"] == "efficientnet"
    assert model_spec["efficientnet_variant"] == "b0"
    assert model_spec["audio_config"]["sample_rate"] == 16000


def test_evaluation_metadata_with_training_metadata(
    temp_dir: Path, mock_config: RunConfig, mock_metrics: Dict[str, float]
) -> None:
    """Test evaluation metadata with training metadata."""
    # Create training metadata DataFrame
    import pandas as pd

    training_metadata = pd.DataFrame(
        [
            {
                "end_timestamp": "2023-01-01T00:00:00",
                "checkpoint_name": "training_checkpoint.pt",
                "is_best": True,
                "is_final": True,
                "config": '{"model_spec": {"name": "efficientnet"}}',
                "model_spec": '{"name": "efficientnet", "efficientnet_variant": "b0"}',
                "output_dir": "/path/to/output",
                "preprocessing": "test_preprocessing",
                "sr": 16000,
                "logging": "test_logging",
                "label_type": "test_label_type",
                "resume_from_checkpoint": None,
                "distributed": False,
                "distributed_backend": "nccl",
                "distributed_port": 12345,
                "augmentations": "test_augmentations",
                "loss_function": "cross_entropy",
                "multilabel": False,
                "run_name": "test_run",
                "wandb_project": "test_project",
                "scheduler": "test_scheduler",
                "debug_mode": False,
                # These should be excluded
                "device": "cpu",
                "seed": 42,
                "num_workers": 0,
                "eval_modes": "test_modes",
                "overwrite_embeddings": False,
            }
        ]
    )

    # Save evaluation metadata with training metadata
    save_evaluation_metadata(
        output_dir=temp_dir,
        dataset_name="test_dataset",
        experiment_name="test_experiment",
        checkpoint_name="test_checkpoint.pt",
        train_metrics=mock_metrics,
        val_metrics=mock_metrics,
        probe_test_metrics=mock_metrics,
        retrieval_metrics=mock_metrics,
        clustering_metrics=mock_metrics,
        eval_config=mock_config.model_dump(mode="json"),
        training_metadata=training_metadata,
        run_config=mock_config.model_dump(mode="json"),
    )

    # Load metadata
    df = load_evaluation_metadata(temp_dir)

    # Verify metadata
    assert not df.empty
    assert "training_params" in df.columns
    assert "run_config_params" in df.columns

    # Verify training parameters are stored as JSON string and exclude unwanted columns
    training_params = get_training_params_from_metadata(df)
    assert training_params is not None
    assert "model_spec" in training_params
    assert "output_dir" in training_params
    assert "preprocessing" in training_params
    assert "sr" in training_params
    assert "logging" in training_params
    assert "label_type" in training_params
    assert "resume_from_checkpoint" in training_params
    assert "distributed" in training_params
    assert "distributed_backend" in training_params
    assert "distributed_port" in training_params
    assert "augmentations" in training_params
    assert "loss_function" in training_params
    assert "multilabel" in training_params
    assert "run_name" in training_params
    assert "wandb_project" in training_params
    assert "scheduler" in training_params
    assert "debug_mode" in training_params

    # Verify excluded columns are not present
    # assert "device" not in training_params
    # assert "seed" not in training_params
    # assert "num_workers" not in training_params
    # assert "eval_modes" not in training_params
    # assert "overwrite_embeddings" not in training_params

    # Verify run config parameters are stored as JSON string
    run_config_params = get_run_config_params_from_metadata(df)
    assert run_config_params is not None
    assert "model_spec" in run_config_params
    assert "training_params" in run_config_params
    assert "dataset_config" in run_config_params
    assert "output_dir" in run_config_params
    assert "loss_function" in run_config_params
    assert "scheduler" in run_config_params
    assert "run_name" in run_config_params
    assert "wandb_project" in run_config_params

    # # Verify excluded columns are not present in run config
    # assert "device" not in run_config_params
    # assert "seed" not in run_config_params
    # assert "num_workers" not in run_config_params
    # assert "eval_modes" not in run_config_params
    # assert "overwrite_embeddings" not in run_config_params


def test_empty_metrics_handling(temp_dir: Path, mock_config: RunConfig) -> None:
    """Test handling of empty metrics."""
    # Save metadata with empty metrics
    save_experiment_metadata(
        output_dir=temp_dir,
        config=mock_config,
        checkpoint_name="test_checkpoint.pt",
        metrics={},
        is_best=True,
        is_final=False,
    )

    # Load metadata
    df = load_experiment_metadata(temp_dir)

    # Verify metadata
    assert not df.empty
    assert "loss" not in df.columns
    assert "accuracy" not in df.columns
    assert "val_loss" not in df.columns
    assert "val_accuracy" not in df.columns
    assert df["checkpoint_name"].iloc[0] == "test_checkpoint.pt"
    assert df["is_best"].iloc[0]
    assert not df["is_final"].iloc[0]

    # Verify model spec fields
    model_spec = json.loads(df["config"].iloc[0])["model_spec"]
    assert model_spec["name"] == "efficientnet"
    assert model_spec["efficientnet_variant"] == "b0"
    assert model_spec["audio_config"]["sample_rate"] == 16000
