"""Integration test for run_evaluate.py with esp-data package."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch.utils.data import Dataset


# Create a minimal dummy dataset that mimics esp_data Dataset behavior
class DummyEspDataset(Dataset):
    """Minimal dataset that mimics esp_data Dataset structure."""

    def __init__(self, n_samples: int = 16):
        self.n_samples = n_samples
        self.data = []
        for i in range(n_samples):
            self.data.append(
                {
                    "raw_wav": torch.randn(16000),  # 1 second of audio at 16kHz
                    "label": torch.randint(0, 3, (1,)).item(),  # Random label 0-2
                    "path": f"/dummy/path_{i}.wav",
                }
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]

    def apply_transformations(self, transformations):
        """Mock apply_transformations method."""
        # Simulate applying label_from_feature transformation
        return {
            "label_from_feature": {
                "label_map": {0: 0, 1: 1, 2: 2},
                "label_feature": "label",
                "num_classes": 3,
            }
        }


class DummyModel(torch.nn.Module):
    """Dummy model that mimics the expected interface."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(16000, 512), torch.nn.ReLU(), torch.nn.Linear(512, 128)
        )
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if isinstance(x, dict):
            x = x["raw_wav"]
        features = self.backbone(x)
        return self.classifier(features)

    def extract_embeddings(
        self, x: torch.Tensor, layers: str = "backbone"
    ) -> torch.Tensor:
        if isinstance(x, dict):
            x = x["raw_wav"]
        return self.backbone(x)


def create_dummy_benchmark_config():
    """Create a dummy benchmark config (old format) that mimics benchmark_base.yml."""
    return {
        "data_path": "benchmark",
        "datasets": [
            {
                "dataset_name": "beans",
                "split": "dogs_test",
                "type": "classification",
                "label_column": "label",
                "audio_path_col": "path",
                "multi_label": False,
                "label_type": "supervised",
                "audio_max_length_seconds": 10,
                "transformations": [
                    {
                        "type": "label_from_feature",
                        "feature": "label",
                        "override": True,
                        "label_map": {0: 0, 1: 1, 2: 2},
                    }
                ],
                "sample_rate": 16000,
                "metrics": ["accuracy", "balanced_accuracy"],
            }
        ],
    }


def create_dummy_structured_config():
    """Create a dummy structured config (new format) like benchmark_structured.yml."""
    base_dataset = {
        "dataset_name": "beans",
        "split": "dogs_test",  # Will be modified for train/val
        "type": "classification",
        "label_column": "label",
        "audio_path_col": "path",
        "multi_label": False,
        "label_type": "supervised",
        "audio_max_length_seconds": 10,
        "transformations": [
            {
                "type": "label_from_feature",
                "feature": "label",
                "override": True,
                "label_map": {0: 0, 1: 1, 2: 2},
            }
        ],
        "sample_rate": 16000,
        "metrics": ["accuracy", "balanced_accuracy"],
    }

    train_dataset = base_dataset.copy()
    train_dataset["split"] = "dogs_train"

    val_dataset = base_dataset.copy()
    val_dataset["split"] = "dogs_val"

    test_dataset = base_dataset.copy()
    test_dataset["split"] = "dogs_test"

    return {
        "train_datasets": [train_dataset],
        "val_datasets": [val_dataset],
        "test_datasets": [test_dataset],
    }


def create_dummy_run_config():
    """Create a dummy run config."""
    return {
        "model_spec": {
            "name": "efficientnet",
            "pretrained": False,
            "audio_config": {
                "sample_rate": 16000,
                "n_mels": 128,
                "representation": "mel_spectrogram",
            },
        },
        "training_params": {
            "train_epochs": 10,
            "lr": 0.001,
            "batch_size": 4,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "amp": False,
            "amp_dtype": "bf16",
        },
        "dataset_config": "dummy_dataset.yml",
        "output_dir": "/tmp",
        "loss_function": "cross_entropy",
    }


def create_dummy_evaluate_config(temp_dir: Path, config_type: str = "benchmark"):
    """Create a dummy evaluate config similar to test.yml."""
    config_file = (
        "dummy_benchmark.yml" if config_type == "benchmark" else "dummy_structured.yml"
    )

    return {
        "dataset_config": str(temp_dir / config_file),
        "training_params": {
            "train_epochs": 2,
            "lr": 0.0003,
            "batch_size": 4,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "amp": False,
            "amp_dtype": "bf16",
        },
        "experiments": [
            {
                "run_name": "dummy_test",
                "run_config": str(temp_dir / "dummy_run.yml"),
                "layers": "backbone",
                "pretrained": False,
                "checkpoint_path": None,  # No checkpoint needed for this test
                "frozen": True,
            }
        ],
        "save_dir": str(temp_dir / "evaluation_results"),
        "device": "cpu",
        "seed": 42,
        "num_workers": 0,
        "eval_modes": ["retrieval"],  # Only test retrieval to avoid training complexity
        "overwrite_embeddings": True,
    }


def mock_dataset_from_config(config):
    """Mock dataset_from_config from esp_data."""
    dataset = DummyEspDataset()
    metadata = {"label_map": {0: 0, 1: 1, 2: 2}, "num_classes": 3}
    return dataset, metadata


def mock_get_model(model_spec, num_classes=3):
    """Mock get_model function."""
    return DummyModel(num_classes)


def test_run_evaluate_integration_with_esp_data_benchmark_format(monkeypatch):
    """Test run_evaluate.py integration with esp-data package using benchmark format."""

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dummy config files - use old benchmark format
        benchmark_config = create_dummy_benchmark_config()
        run_config = create_dummy_run_config()
        evaluate_config = create_dummy_evaluate_config(temp_path, "benchmark")

        # Write config files
        benchmark_file = temp_path / "dummy_benchmark.yml"
        run_file = temp_path / "dummy_run.yml"
        eval_file = temp_path / "dummy_eval.yml"

        with open(benchmark_file, "w") as f:
            yaml.dump(benchmark_config, f)
        with open(run_file, "w") as f:
            yaml.dump(run_config, f)
        with open(eval_file, "w") as f:
            yaml.dump(evaluate_config, f)

        # Mock esp_data functions
        import representation_learning.run_evaluate as run_eval_module
        from representation_learning.data import dataset as dataset_module
        from representation_learning.models import get_model as get_model_module

        # Mock dataset_from_config from esp_data
        monkeypatch.setattr("esp_data.dataset_from_config", mock_dataset_from_config)

        # Mock get_model
        monkeypatch.setattr(get_model_module, "get_model", mock_get_model)
        monkeypatch.setattr(run_eval_module, "get_model", mock_get_model)

        # Mock build_dataloaders to return our dummy datasets
        def mock_build_dataloaders(cfg, data_config, device):
            from torch.utils.data import DataLoader

            dummy_ds = DummyEspDataset()
            dummy_ds.metadata = {"label_map": {0: 0, 1: 1, 2: 2}}
            dummy_dl = DataLoader(dummy_ds, batch_size=2, shuffle=False)
            return None, None, dummy_dl  # Only return test dataloader

        monkeypatch.setattr(dataset_module, "build_dataloaders", mock_build_dataloaders)
        monkeypatch.setattr(
            run_eval_module, "build_dataloaders", mock_build_dataloaders
        )

        # Mock anypath to handle local paths
        def mock_anypath(path):
            return Path(path)

        monkeypatch.setattr("esp_data.io.anypath", mock_anypath)
        monkeypatch.setattr(run_eval_module, "anypath", mock_anypath)

        # Import and run the evaluation
        from representation_learning.configs import load_config
        from representation_learning.run_evaluate import run_experiment

        # Load configs - use correct types
        eval_cfg = load_config(eval_file, config_type="evaluate")
        benchmark_cfg = load_config(
            benchmark_file, config_type="benchmark"
        )  # Old format

        # Convert to collection format
        data_collection_cfg = benchmark_cfg.to_dataset_collection_config()

        # Run experiment
        device = torch.device("cpu")

        # Test the experiment
        for ds_cfg in data_collection_cfg.test_datasets:
            for exp_cfg in eval_cfg.experiments:
                result = run_experiment(
                    eval_cfg=eval_cfg,
                    dataset_cfg=ds_cfg,
                    experiment_cfg=exp_cfg,
                    data_collection_cfg=data_collection_cfg,
                    device=device,
                    save_dir=temp_path / "results",
                )

                # Verify result structure
                assert result is not None
                assert hasattr(result, "dataset_name")
                assert hasattr(result, "experiment_name")
                assert hasattr(result, "retrieval_metrics")
                assert isinstance(result.retrieval_metrics, dict)


def test_run_evaluate_integration_with_esp_data_structured_format(monkeypatch):
    """Test run_evaluate.py integration with esp-data package using structured format."""

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dummy config files - use new structured format
        structured_config = create_dummy_structured_config()
        run_config = create_dummy_run_config()
        evaluate_config = create_dummy_evaluate_config(temp_path, "structured")

        # Write config files
        structured_file = temp_path / "dummy_structured.yml"
        run_file = temp_path / "dummy_run.yml"
        eval_file = temp_path / "dummy_eval.yml"

        with open(structured_file, "w") as f:
            yaml.dump(structured_config, f)
        with open(run_file, "w") as f:
            yaml.dump(run_config, f)
        with open(eval_file, "w") as f:
            yaml.dump(evaluate_config, f)

        # Mock esp_data functions
        import representation_learning.run_evaluate as run_eval_module
        from representation_learning.data import dataset as dataset_module
        from representation_learning.models import get_model as get_model_module

        # Mock dataset_from_config from esp_data
        monkeypatch.setattr("esp_data.dataset_from_config", mock_dataset_from_config)

        # Mock get_model
        monkeypatch.setattr(get_model_module, "get_model", mock_get_model)
        monkeypatch.setattr(run_eval_module, "get_model", mock_get_model)

        # Mock build_dataloaders to return our dummy datasets
        def mock_build_dataloaders(cfg, data_config, device):
            from torch.utils.data import DataLoader

            dummy_ds = DummyEspDataset()
            dummy_ds.metadata = {"label_map": {0: 0, 1: 1, 2: 2}}
            dummy_dl = DataLoader(dummy_ds, batch_size=2, shuffle=False)
            return None, None, dummy_dl  # Only return test dataloader

        monkeypatch.setattr(dataset_module, "build_dataloaders", mock_build_dataloaders)
        monkeypatch.setattr(
            run_eval_module, "build_dataloaders", mock_build_dataloaders
        )

        # Mock anypath to handle local paths
        def mock_anypath(path):
            return Path(path)

        monkeypatch.setattr("esp_data.io.anypath", mock_anypath)
        monkeypatch.setattr(run_eval_module, "anypath", mock_anypath)

        # Import and run the evaluation
        from representation_learning.configs import load_config
        from representation_learning.run_evaluate import run_experiment

        # Load configs - use structured format as DatasetCollectionConfig
        eval_cfg = load_config(eval_file, config_type="evaluate")
        data_collection_cfg = load_config(
            structured_file, config_type="data"
        )  # New format

        # Run experiment
        device = torch.device("cpu")

        # Test the experiment
        for ds_cfg in data_collection_cfg.test_datasets:
            for exp_cfg in eval_cfg.experiments:
                result = run_experiment(
                    eval_cfg=eval_cfg,
                    dataset_cfg=ds_cfg,
                    experiment_cfg=exp_cfg,
                    data_collection_cfg=data_collection_cfg,
                    device=device,
                    save_dir=temp_path / "results",
                )

                # Verify result structure
                assert result is not None
                assert hasattr(result, "dataset_name")
                assert hasattr(result, "experiment_name")
                assert hasattr(result, "retrieval_metrics")
                assert isinstance(result.retrieval_metrics, dict)


def test_esp_data_label_transform_patch():
    """Test that the esp_data pandas indexing patch is working correctly."""

    # Test the patched LabelFromFeature functionality
    try:
        import pandas as pd
        from esp_data.transforms.label_from_feature import LabelFromFeature

        # Create test data
        df = pd.DataFrame(
            {"label_text": ["cat", "dog", "bird", "cat"], "other_col": [1, 2, 3, 4]}
        )

        # Create transform with label map
        transform = LabelFromFeature(
            feature="label_text",
            output_feature="label",
            label_map={"cat": 0, "dog": 1, "bird": 2},
            override=True,
        )

        # Apply transform - this should not raise the pandas broadcasting error
        result_df, metadata = transform(df)

        # Verify the transform worked
        assert "label" in result_df.columns
        assert list(result_df["label"]) == [0, 1, 2, 0]
        assert metadata["num_classes"] == 3

    except ImportError:
        # If esp_data is not available, skip this test
        pytest.skip("esp_data not available for testing")


def test_config_format_compatibility():
    """Test that both old and new config formats work."""

    # Test old format (benchmark_base.yml style)
    old_config = create_dummy_benchmark_config()

    # Test new format (benchmark_structured.yml style)
    new_config = create_dummy_structured_config()

    # Both should be valid when parsed with correct config types
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        old_file = temp_path / "old_config.yml"
        new_file = temp_path / "new_config.yml"

        with open(old_file, "w") as f:
            yaml.dump(old_config, f)
        with open(new_file, "w") as f:
            yaml.dump(new_config, f)

        from representation_learning.configs import load_config

        # Load with correct config types
        old_cfg = load_config(old_file, config_type="benchmark")  # Old format
        new_cfg = load_config(new_file, config_type="data")  # New format

        assert old_cfg is not None
        assert new_cfg is not None

        # Old format should have datasets
        assert hasattr(old_cfg, "datasets")
        assert len(old_cfg.datasets) > 0

        # New format should have train/val/test splits
        assert hasattr(new_cfg, "train_datasets")
        assert hasattr(new_cfg, "val_datasets")
        assert hasattr(new_cfg, "test_datasets")

        # Test conversion from old to new format
        converted_cfg = old_cfg.to_dataset_collection_config()
        assert hasattr(converted_cfg, "train_datasets")
        assert hasattr(converted_cfg, "val_datasets")
        assert hasattr(converted_cfg, "test_datasets")


if __name__ == "__main__":
    # Run the tests
    test_esp_data_label_transform_patch()
    test_config_format_compatibility()
    print("All integration tests passed!")
