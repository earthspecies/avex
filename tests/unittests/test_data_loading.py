"""
Unit tests for data loading and processing.
"""

from pathlib import Path

import pandas as pd
import yaml

from representation_learning.configs import (
    AudioConfig,
    RunConfig,
    TrainingParams,
)
from representation_learning.data.data_utils import build_dataloaders
from representation_learning.data.dataset import AudioDataset, get_dataset_dummy


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
        "subset_percentage": 1.0,
        "label_column": "canonical_name",
        "label_type": "supervised",
        "audio_max_length_seconds": 10,
        "transformations": [
            {
                "filter": {
                    "property": "source",
                    "values": ["xeno-canto", "iNaturalist"],
                    "operation": "include",
                }
            }
        ],
    }

    # Save data config
    data_config_path = tmp_path / "data_config.yml"
    with open(data_config_path, "w") as f:
        yaml.dump(data_config, f)

    # Create run config
    run_config = RunConfig(
        model_config={"name": "efficientnetb0", "pretrained": True},
        dataset_config=str(data_config_path),
        preprocessing=None,
        sr=16000,
        logging="mlflow",
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
        training_params=TrainingParams(
            train_epochs=10,
            lr=0.0001,
            batch_size=16,
            optimizer="adamw",
            weight_decay=0.01,
        ),
        augmentations=[],
        loss_function="cross_entropy",
        device="cpu",
        seed=42,
        num_workers=0,
        run_name="test",
        wandb_project="test",
    )

    return run_config


def test_get_dataset_dummy(tmp_path: Path) -> None:
    """Test loading a dataset with transformations.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path for test files
    """
    # Create test data
    csv_path = create_test_csv(tmp_path)

    # Create test config
    run_config = create_test_config(tmp_path, csv_path)

    # Load dataset
    dataset = get_dataset_dummy(
        data_config=run_config.dataset_config, transform=None, preprocessor=None
    )

    # Check dataset properties
    assert isinstance(dataset, AudioDataset)
    assert len(dataset) == 10  # All samples should be included
    assert set(dataset.metadata["source"]) == {"xeno-canto", "iNaturalist"}
    assert set(dataset.metadata["canonical_name"]) == {"bird", "mammal"}


def test_build_dataloaders(tmp_path: Path) -> None:
    """Test building dataloaders from configuration.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path for test files
    """
    # Create test data
    csv_path = create_test_csv(tmp_path)

    # Create test config
    run_config = create_test_config(tmp_path, csv_path)

    # Build dataloaders
    train_dl, val_dl = build_dataloaders(run_config, device="cpu")

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
    assert batch["raw_wav"].shape[1] == run_config.audio_config.target_length


def test_dataset_transforms(tmp_path: Path) -> None:
    """Test dataset with transformations.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path for test files
    """
    # Create test data
    csv_path = create_test_csv(tmp_path)

    # Create test config with subsampling
    run_config = create_test_config(tmp_path, csv_path)
    with open(run_config.dataset_config, "r") as f:
        data_config = yaml.safe_load(f)
    data_config["transformations"].append(
        {
            "subsample": {
                "property": "class",
                "operation": "subsample",
                "ratios": {"birds": 0.5, "mammals": 0.5},
            }
        }
    )
    with open(run_config.dataset_config, "w") as f:
        yaml.dump(data_config, f)

    # Load dataset
    dataset = get_dataset_dummy(
        data_config=run_config.dataset_config, transform=None, preprocessor=None
    )

    # Check that subsampling was applied
    class_counts = dataset.metadata["class"].value_counts()
    assert abs(class_counts["birds"] / 5 - 0.5) < 0.1
    assert abs(class_counts["mammals"] / 5 - 0.5) < 0.1
