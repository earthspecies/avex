"""
Integration test for dataset label transformation fix.

This test verifies that:
1. Dataloaders can be built successfully for train/val/test splits
2. Label maps are correctly propagated to all splits
3. Label transformation is skipped when labels are already integers
4. Batch creation works without overflow errors
5. Labels are in the correct format (integers for single-label, lists for multi-label)
"""

import math
from pathlib import Path
from typing import Callable, Optional

import pytest
import torch

from representation_learning.configs import (
    AudioConfig,
    DatasetCollectionConfig,
    EvaluateConfig,
    ModelSpec,
    RunConfig,
    SchedulerConfig,
    TrainingParams,
)
from representation_learning.data.dataset import build_dataloaders


class TestDatasetLabelTransformation:
    @pytest.fixture
    def config_path(self) -> Path:
        return Path("configs/evaluation_configs/cpu_test.yml")

    @pytest.fixture
    def minimal_run_config(self) -> Callable[..., RunConfig]:
        """Create a minimal run config for testing.

        Returns
        -------
        Callable
            A function that creates RunConfig instances with the specified parameters.
        """

        def _create_config(
            sample_rate: int = 16000,
            audio_max_length_seconds: int = 10,
            dataset_config: Optional[DatasetCollectionConfig] = None,
        ) -> RunConfig:
            audio_cfg = AudioConfig(
                sample_rate=sample_rate,
                target_length_seconds=audio_max_length_seconds,
                representation="raw",
            )
            model_spec = ModelSpec(
                name="efficientnet",
                pretrained=False,
                device="cpu",
                audio_config=audio_cfg,
                efficientnet_variant="b0",
            )
            training_params = TrainingParams(
                train_epochs=1,
                lr=0.001,
                batch_size=2,
                optimizer="adamw",
                weight_decay=0.0,
                amp=False,
                amp_dtype="bf16",
                log_steps=1,
                gradient_checkpointing=False,
            )
            scheduler = SchedulerConfig(name="none")
            return RunConfig(
                model_spec=model_spec,
                training_params=training_params,
                dataset_config=dataset_config,
                output_dir="/tmp",
                sr=sample_rate,
                logging="none",
                label_type="supervised",
                augmentations=[],
                loss_function="cross_entropy",
                multilabel=False,
                device="cpu",
                seed=42,
                num_workers=2,
                run_name="test_run",
                wandb_project="test-project",
                scheduler=scheduler,
                debug_mode=False,
            )

        return _create_config

    def test_dataloader_creation_success(self, config_path: Path, minimal_run_config: Callable[..., RunConfig]) -> None:
        """Test that dataloaders can be created without errors."""
        # Load evaluation config
        eval_cfg = EvaluateConfig.from_sources(yaml_file=config_path, cli_args=())

        # Get dataset config
        benchmark_eval_cfg = eval_cfg.dataset_config
        evaluation_sets = benchmark_eval_cfg.get_all_evaluation_sets()
        assert evaluation_sets, "No evaluation sets found"

        # Use first evaluation set
        eval_set_name, eval_set_data_cfg = evaluation_sets[0]
        test_datasets = eval_set_data_cfg.test_datasets or []
        assert test_datasets, f"No test datasets in evaluation set '{eval_set_name}'"
        test_ds_cfg = test_datasets[0]

        # Create data collection config
        data_collection_cfg = DatasetCollectionConfig(
            train_datasets=eval_set_data_cfg.train_datasets,
            val_datasets=eval_set_data_cfg.val_datasets,
            test_datasets=[test_ds_cfg],
        )

        # Create run config
        sample_rate = getattr(test_ds_cfg, "sample_rate", 16000)
        audio_max_length_seconds = getattr(test_ds_cfg, "audio_max_length_seconds", 10)
        run_cfg = minimal_run_config(sample_rate, audio_max_length_seconds, data_collection_cfg)

        # Build dataloaders - this should not raise any errors
        train_dl, val_dl, test_dl = build_dataloaders(
            run_cfg,
            data_config=data_collection_cfg,
            device="cpu",
            is_evaluation_context=True,
        )

        # Basic assertions
        assert train_dl is not None, "Train dataloader should not be None"
        assert val_dl is not None, "Validation dataloader should not be None"
        assert test_dl is not None, "Test dataloader should not be None"

    def test_label_map_propagation(self, config_path: Path, minimal_run_config: Callable[..., RunConfig]) -> None:
        """Test that label maps are correctly propagated to all splits."""
        # Load evaluation config
        eval_cfg = EvaluateConfig.from_sources(yaml_file=config_path, cli_args=())
        benchmark_eval_cfg = eval_cfg.dataset_config
        evaluation_sets = benchmark_eval_cfg.get_all_evaluation_sets()
        eval_set_name, eval_set_data_cfg = evaluation_sets[0]
        test_ds_cfg = eval_set_data_cfg.test_datasets[0]

        # Create data collection config and run config
        data_collection_cfg = DatasetCollectionConfig(
            train_datasets=eval_set_data_cfg.train_datasets,
            val_datasets=eval_set_data_cfg.val_datasets,
            test_datasets=[test_ds_cfg],
        )
        run_cfg = minimal_run_config(dataset_config=data_collection_cfg)

        # Build dataloaders
        train_dl, val_dl, test_dl = build_dataloaders(
            run_cfg,
            data_config=data_collection_cfg,
            device="cpu",
            is_evaluation_context=True,
        )

        # Check that all datasets have metadata
        assert hasattr(train_dl.dataset, "metadata"), "Train dataset missing metadata"
        assert hasattr(val_dl.dataset, "metadata"), "Val dataset missing metadata"
        assert hasattr(test_dl.dataset, "metadata"), "Test dataset missing metadata"

        # Check that label maps are present and consistent
        train_label_map = train_dl.dataset.metadata.get("label_map", {})
        val_label_map = val_dl.dataset.metadata.get("label_map", {})
        test_label_map = test_dl.dataset.metadata.get("label_map", {})

        assert train_label_map, "Train dataset should have non-empty label_map"
        assert val_label_map, "Val dataset should have non-empty label_map"
        assert test_label_map, "Test dataset should have non-empty label_map"

        # Label maps should be consistent across splits
        assert train_label_map == val_label_map, "Train and val label maps should be identical"
        assert train_label_map == test_label_map, "Train and test label maps should be identical"

        # Check num_labels consistency
        train_num_labels = train_dl.dataset.metadata.get("num_labels", 0)
        val_num_labels = val_dl.dataset.metadata.get("num_labels", 0)
        test_num_labels = test_dl.dataset.metadata.get("num_labels", 0)

        assert train_num_labels > 1, "Should have multiple labels for classification"
        assert train_num_labels == val_num_labels == test_num_labels, "num_labels should be consistent"

    def test_label_format_consistency(self, config_path: Path, minimal_run_config: Callable[..., RunConfig]) -> None:
        """Test that labels are in the correct format across all splits."""
        # Load evaluation config and build dataloaders
        eval_cfg = EvaluateConfig.from_sources(yaml_file=config_path, cli_args=())
        benchmark_eval_cfg = eval_cfg.dataset_config
        evaluation_sets = benchmark_eval_cfg.get_all_evaluation_sets()
        eval_set_name, eval_set_data_cfg = evaluation_sets[0]
        test_ds_cfg = eval_set_data_cfg.test_datasets[0]

        data_collection_cfg = DatasetCollectionConfig(
            train_datasets=eval_set_data_cfg.train_datasets,
            val_datasets=eval_set_data_cfg.val_datasets,
            test_datasets=[test_ds_cfg],
        )
        run_cfg = minimal_run_config(dataset_config=data_collection_cfg)

        train_dl, val_dl, test_dl = build_dataloaders(
            run_cfg,
            data_config=data_collection_cfg,
            device="cpu",
            is_evaluation_context=True,
        )

        # Check label format in individual samples
        for split_name, dataloader in [
            ("train", train_dl),
            ("val", val_dl),
            ("test", test_dl),
        ]:
            if len(dataloader.dataset) > 0:
                sample = dataloader.dataset[0]
                label = sample.get("label")

                # Labels should be integers (for single-label classification)
                assert isinstance(label, int), (
                    f"{split_name} sample label should be integer, got {type(label)}: {label}"
                )
                assert label >= 0, f"{split_name} label should be non-negative, got {label}"

                # Check that label is within expected range
                num_labels = dataloader.dataset.metadata.get("num_labels", 0)
                assert 0 <= label < num_labels, f"{split_name} label {label} out of range [0, {num_labels})"

    def test_batch_creation_no_overflow(self, config_path: Path, minimal_run_config: Callable[..., RunConfig]) -> None:
        """Test that batches can be created without overflow errors."""
        # Load evaluation config and build dataloaders
        eval_cfg = EvaluateConfig.from_sources(yaml_file=config_path, cli_args=())
        benchmark_eval_cfg = eval_cfg.dataset_config
        evaluation_sets = benchmark_eval_cfg.get_all_evaluation_sets()
        eval_set_name, eval_set_data_cfg = evaluation_sets[0]
        test_ds_cfg = eval_set_data_cfg.test_datasets[0]

        data_collection_cfg = DatasetCollectionConfig(
            train_datasets=eval_set_data_cfg.train_datasets,
            val_datasets=eval_set_data_cfg.val_datasets,
            test_datasets=[test_ds_cfg],
        )
        run_cfg = minimal_run_config(dataset_config=data_collection_cfg)

        train_dl, val_dl, test_dl = build_dataloaders(
            run_cfg,
            data_config=data_collection_cfg,
            device="cpu",
            is_evaluation_context=True,
        )

        # Test batch creation for each split
        for split_name, dataloader in [
            ("train", train_dl),
            ("val", val_dl),
            ("test", test_dl),
        ]:
            try:
                batch = next(iter(dataloader))

                # Check batch structure
                assert "raw_wav" in batch, f"{split_name} batch missing 'raw_wav'"
                assert "label" in batch, f"{split_name} batch missing 'label'"
                assert "padding_mask" in batch, f"{split_name} batch missing 'padding_mask'"

                # Check label tensor properties
                label_tensor = batch["label"]
                assert isinstance(label_tensor, torch.Tensor), f"{split_name} labels should be tensor"
                assert label_tensor.dtype == torch.float32, f"{split_name} labels should be float32"
                assert len(label_tensor.shape) == 2, f"{split_name} labels should be 2D (batch_size, num_classes)"

                # Check that labels are one-hot encoded
                assert torch.all((label_tensor == 0) | (label_tensor == 1)), f"{split_name} labels should be one-hot"
                assert torch.all(label_tensor.sum(dim=1) == 1), (
                    f"{split_name} each sample should have exactly one label"
                )

                print(f"✅ {split_name.capitalize()} batch creation successful")

            except RuntimeError as e:
                if "overflow" in str(e).lower():
                    pytest.fail(f"{split_name} batch creation failed with overflow error: {e}")
                else:
                    raise e

    def test_no_nan_labels(self, config_path: Path, minimal_run_config: Callable[..., RunConfig]) -> None:
        """Test that no labels are NaN after transformation."""
        # Load evaluation config and build dataloaders
        eval_cfg = EvaluateConfig.from_sources(yaml_file=config_path, cli_args=())
        benchmark_eval_cfg = eval_cfg.dataset_config
        evaluation_sets = benchmark_eval_cfg.get_all_evaluation_sets()
        eval_set_name, eval_set_data_cfg = evaluation_sets[0]
        test_ds_cfg = eval_set_data_cfg.test_datasets[0]

        data_collection_cfg = DatasetCollectionConfig(
            train_datasets=eval_set_data_cfg.train_datasets,
            val_datasets=eval_set_data_cfg.val_datasets,
            test_datasets=[test_ds_cfg],
        )
        run_cfg = minimal_run_config(dataset_config=data_collection_cfg)

        train_dl, val_dl, test_dl = build_dataloaders(
            run_cfg,
            data_config=data_collection_cfg,
            device="cpu",
            is_evaluation_context=True,
        )

        # Check first few samples from each split for NaN labels

        for split_name, dataloader in [
            ("train", train_dl),
            ("val", val_dl),
            ("test", test_dl),
        ]:
            dataset = dataloader.dataset
            samples_to_check = min(5, len(dataset))

            for i in range(samples_to_check):
                sample = dataset[i]
                label = sample.get("label")

                # Check for NaN
                if isinstance(label, float) and math.isnan(label):
                    pytest.fail(f"{split_name} sample {i} has NaN label")
                elif isinstance(label, list) and any(math.isnan(x) for x in label if isinstance(x, float)):
                    pytest.fail(f"{split_name} sample {i} has NaN in label list")

                # Ensure label is not None
                assert label is not None, f"{split_name} sample {i} has None label"
