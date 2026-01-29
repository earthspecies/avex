"""
Unit tests for data loading and processing.

These tests require esp_data which is an internal dependency.
They are skipped when esp_data is not installed.
"""

from pathlib import Path

import pandas as pd
import pytest

from representation_learning.configs import (
    AudioConfig,
    DatasetCollectionConfig,
    ModelSpec,
    RunConfig,
    TrainingParams,
)
from representation_learning.data.dataset import (
    build_dataloaders,
)

# Skip entire module if esp_data is not installed (internal dependency)
esp_data = pytest.importorskip("esp_data")
Dataset = esp_data.Dataset
DatasetConfig = esp_data.DatasetConfig
dataset_from_config = esp_data.dataset_from_config


def create_test_csv(tmp_path: Path) -> Path:
    """Create a test CSV file with sample data.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path for creating test files

    Returns
    -------
    Path
        Path to the created CSV file
    """
    df = pd.DataFrame(
        {
            "filepath": [str(tmp_path / f"test_{i}.wav") for i in range(10)],
            "canonical_name": ["bird"] * 5 + ["mammal"] * 5,
            "source": ["xeno-canto"] * 5 + ["iNaturalist"] * 5,
            "class": ["birds"] * 5 + ["mammals"] * 5,
        }
    )

    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    # Create dummy audio files
    for i in range(10):
        audio_path = tmp_path / f"test_{i}.wav"
        # Create empty file for testing
        audio_path.touch()

    return csv_path


def create_test_config(tmp_path: Path, csv_path: Path) -> RunConfig:
    """Create a test configuration.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path for creating config files
    csv_path : Path
        Path to the CSV file containing test data

    Returns
    -------
    RunConfig
        Test run configuration
    """
    data_config = {
        "dataset_name": "test",
        "source_dataset_name": "test",
        "dataset_version": "0.0",
        "dataset_source": str(csv_path),
        "balance": False,
        "balance_attribute": "canonical_name",
        "custom_balancing": False,
        "balancing_method": "upsample",
        "label_column": "canonical_name",
        "label_type": "supervised",
        "audio_max_length_seconds": 10,
        # Avoid esp_data transformation schema coupling in unit test
    }

    # Build DatasetCollectionConfig directly
    ds_cfg = DatasetConfig(**data_config)
    ds_collection = DatasetCollectionConfig(
        train_datasets=[ds_cfg],
        val_datasets=[ds_cfg],
        test_datasets=None,
        transformations=None,
    )

    # Create run config (match current schema)
    run_config = RunConfig(
        model_spec=ModelSpec(
            name="efficientnet",
            pretrained=True,
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
        ),
        training_params=TrainingParams(
            train_epochs=1,
            lr=0.0001,
            batch_size=4,
            optimizer="adamw",
            weight_decay=0.01,
        ),
        dataset_config=ds_collection,
        output_dir=str(tmp_path),
        sr=16000,
        num_workers=0,
        loss_function="cross_entropy",
        run_name="test",
    )

    return run_config


def test_load_dataset_from_yaml(tmp_path: Path) -> None:
    """Test loading a dataset directly from YAML using esp_data API.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path for test files
    """
    # Create test data
    csv_path = create_test_csv(tmp_path)

    # Create test config
    _run_config = create_test_config(tmp_path, csv_path)

    # Load dataset using esp_data; skip if dataset not registered
    test_cfg = {
        "dataset_name": "test",
        "source_dataset_name": "test",
        "dataset_version": "0.0",
        "dataset_source": str(csv_path),
        "balance": False,
        "balance_attribute": "canonical_name",
        "custom_balancing": False,
        "balancing_method": "upsample",
        "label_column": "canonical_name",
        "label_type": "supervised",
        "audio_max_length_seconds": 10,
    }
    ds_cfg = DatasetConfig(**test_cfg)
    try:
        dataset, metadata = dataset_from_config(ds_cfg)
    except KeyError:
        pytest.skip("esp_data registry has no 'test' dataset; skipping")

    # Check dataset properties
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 10  # All samples should be included
    # Metadata is available either on dataset or as a separate return
    meta = getattr(dataset, "metadata", None) or metadata
    assert set(meta["source"]) == {"xeno-canto", "iNaturalist"}
    assert set(meta["canonical_name"]) == {"bird", "mammal"}


def test_build_dataloaders(tmp_path: Path) -> None:
    """Test building dataloaders from configuration.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path for test files

    Raises
    ------
    ValueError
        If the underlying dataset is not registered in esp_data.
    """
    # Create test data
    csv_path = create_test_csv(tmp_path)

    # Create test config
    run_config = create_test_config(tmp_path, csv_path)

    # Build dataloaders; skip if underlying dataset is not registered
    try:
        train_dl, val_dl = build_dataloaders(run_config, device="cpu")
    except ValueError as e:
        if "is not registered" in str(e):
            pytest.skip("esp_data registry missing test dataset; skipping")
        raise

    # Check dataloader properties
    assert len(train_dl) > 0
    assert len(val_dl) > 0
    assert train_dl.batch_size == run_config.training_params.batch_size
    assert val_dl.batch_size == run_config.training_params.batch_size

    # Check a batch
    batch = next(iter(train_dl))
    assert "raw_wav" in batch
    assert "padding_mask" in batch
    assert "label" in batch
    assert batch["raw_wav"].shape[0] <= run_config.training_params.batch_size
    target_len = (
        run_config.model_spec.audio_config.target_length_seconds * run_config.model_spec.audio_config.sample_rate
    )
    assert batch["raw_wav"].shape[1] == target_len


def test_dataset_basic_load(tmp_path: Path) -> None:
    """Basic dataset load without transformations (schema-agnostic)."""
    csv_path = create_test_csv(tmp_path)
    _run_config = create_test_config(tmp_path, csv_path)
    # Build minimal DatasetConfig without transformations
    data_cfg = {
        "dataset_name": "test",
        "source_dataset_name": "test",
        "dataset_version": "0.0",
        "dataset_source": str(csv_path),
        "balance": False,
        "balance_attribute": "canonical_name",
        "custom_balancing": False,
        "balancing_method": "upsample",
        "label_column": "canonical_name",
        "label_type": "supervised",
        "audio_max_length_seconds": 10,
    }
    ds_cfg = DatasetConfig(**data_cfg)
    try:
        dataset, metadata = dataset_from_config(ds_cfg)
    except KeyError:
        pytest.skip("esp_data registry has no 'test' dataset; skipping")
    meta = getattr(dataset, "metadata", None) or metadata
    assert len(dataset) == 10
    assert set(meta["canonical_name"]) == {"bird", "mammal"}
