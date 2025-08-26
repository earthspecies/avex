"""
Unit tests for data loading and processing.
"""

from pathlib import Path

from esp_data import Dataset

from representation_learning.configs import (
    DatasetCollectionConfig,
    RunConfig,
)
from representation_learning.data.dataset import (
    AudioDataset,
    _build_datasets,
    build_dataloaders,
)


def create_test_config() -> RunConfig:
    """Create a test configuration.
    Returns
    -------
    RunConfig
        Test run configuration
    """
    run_config = RunConfig.from_sources(
        "tests/samples/test_run_config.yml", cli_args=()
    )

    return run_config


def test_build_datasets() -> None:
    config_path = "tests/samples/test_data_config.yml"
    cfg = DatasetCollectionConfig.from_sources(config_path, cli_args=())
    train_ds, val_ds, test_ds = _build_datasets(
        cfg, postprocessors=[], label_type="category"
    )

    assert test_ds is None
    assert isinstance(train_ds, AudioDataset)
    assert isinstance(train_ds.ds, Dataset)
    assert isinstance(val_ds, AudioDataset)
    assert len(train_ds) > 0
    assert len(val_ds) > 0

    for sample in train_ds:
        assert isinstance(sample, dict)
        assert "audio" in sample
        assert len(sample["audio"]) > 0
        assert "label" in sample
        break


def test_build_dataloaders(tmp_path: Path) -> None:
    """Test building dataloaders from configuration.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path for test files
    """
    # Create test config
    run_config = create_test_config()

    # Build dataloaders
    train_dl, val_dl, _ = build_dataloaders(
        run_config,
        device="cpu",
        task_type="detection",
        dataset_audio_max_length_seconds=10,
        enable_eval_augmentations=False,
        is_evaluation_context=False,
    )

    # Check dataloader properties
    assert len(train_dl) > 0
    assert len(val_dl) > 0
    assert train_dl.batch_size == run_config.training_params.batch_size
    assert val_dl.batch_size == run_config.training_params.batch_size

    # Check a batch
    batch = next(iter(val_dl))
    assert "raw_wav" in batch
    assert "padding_mask" in batch
    assert "label" in batch
    assert batch["raw_wav"].shape[0] <= run_config.training_params.batch_size
    assert (
        batch["raw_wav"].shape[1]
        == run_config.model_spec.audio_config.target_length_seconds
        * run_config.model_spec.audio_config.sample_rate
    )
