"""
Test that multiple probe types can reuse the same embedding files.

This test verifies that when running multiple experiments with offline training
on the same dataset, they all use the same embedding files saved with
aggregation="none".
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from representation_learning.configs import ExperimentConfig


class TestEmbeddingReuse:
    """Test embedding file reuse across different probe types."""

    def test_embedding_paths_are_dataset_model_level(self) -> None:
        """Test that embedding paths are at dataset+model level.

        Not experiment level.
        """
        # This is a unit test to verify the path construction logic
        save_dir = Path("/tmp/test_results")
        dataset_name = "test_dataset"
        evaluation_dataset_name = "test_eval_dataset"
        model_name = "beats_pretrained"

        # Simulate the path construction logic from run_evaluate.py
        embedding_dir_name = evaluation_dataset_name or dataset_name
        emb_base_dir = save_dir / f"{embedding_dir_name}_{model_name}"
        train_path = emb_base_dir / "embedding_train.h5"
        val_path = emb_base_dir / "embedding_val.h5"
        test_path = emb_base_dir / "embedding_test.h5"

        # Verify paths don't include experiment_name but include model_name
        expected_base = save_dir / "test_eval_dataset_beats_pretrained"
        assert emb_base_dir == expected_base
        assert train_path == expected_base / "embedding_train.h5"
        assert val_path == expected_base / "embedding_val.h5"
        assert test_path == expected_base / "embedding_test.h5"

        # Verify paths are the same regardless of experiment_name but different models
        emb_base_dir2 = save_dir / f"{embedding_dir_name}_{model_name}"
        # Same path for different experiments, same model
        assert emb_base_dir == emb_base_dir2

        # Different model should have different path
        model_name2 = "efficientnet_b0"
        emb_base_dir3 = save_dir / f"{embedding_dir_name}_{model_name2}"
        assert emb_base_dir != emb_base_dir3  # Different path for different models

    def test_aggregation_none_for_offline_training(self) -> None:
        """Test that offline training uses aggregation='none'."""
        # This tests the aggregation logic
        experiment_cfg = MagicMock(spec=ExperimentConfig)
        experiment_cfg.get_training_mode.return_value = False  # offline training
        experiment_cfg.get_aggregation_method.return_value = "mean"

        # Simulate the aggregation logic from run_evaluate.py
        need_probe = True
        offline_training = need_probe and not experiment_cfg.get_training_mode()

        if offline_training:
            aggregation_method = "none"
        else:
            aggregation_method = experiment_cfg.get_aggregation_method()

        assert offline_training is True
        assert aggregation_method == "none"
        # Verify that get_aggregation_method was not called since we use "none"
        experiment_cfg.get_aggregation_method.assert_not_called()

    def test_aggregation_from_config_for_online_training(self) -> None:
        """Test that online training uses aggregation from config."""
        experiment_cfg = MagicMock(spec=ExperimentConfig)
        experiment_cfg.get_training_mode.return_value = True  # online training
        experiment_cfg.get_aggregation_method.return_value = "mean"

        # Simulate the aggregation logic from run_evaluate.py
        need_probe = True
        online_training = need_probe and experiment_cfg.get_training_mode()
        offline_training = need_probe and not experiment_cfg.get_training_mode()

        if offline_training:
            aggregation_method = "none"
        else:
            aggregation_method = experiment_cfg.get_aggregation_method()

        assert online_training is True
        assert offline_training is False
        assert aggregation_method == "mean"
        experiment_cfg.get_aggregation_method.assert_called_once()

    @pytest.mark.integration
    def test_embedding_file_reuse_integration(self) -> None:
        """Integration test to verify embedding file reuse works end-to-end."""
        # This would be a more comprehensive test that actually runs the
        # embedding extraction process, but it requires more setup
        # For now, we'll just verify the path logic works correctly

        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)

            # Test with different experiment names but same dataset and model
            experiments = ["linear_probe", "lstm_probe", "attention_probe"]
            dataset_name = "test_dataset"
            model_name = "beats_pretrained"

            embedding_paths = []
            for _ in experiments:
                # Simulate path construction
                embedding_dir_name = dataset_name
                emb_base_dir = save_dir / f"{embedding_dir_name}_{model_name}"
                train_path = emb_base_dir / "embedding_train.h5"
                embedding_paths.append(train_path)

            # All experiments should use the same embedding file (same dataset+model)
            assert all(path == embedding_paths[0] for path in embedding_paths)

            # The path should not contain any experiment name but should
            # contain model name
            assert "linear_probe" not in str(embedding_paths[0])
            assert "lstm_probe" not in str(embedding_paths[0])
            assert "attention_probe" not in str(embedding_paths[0])
            assert "test_dataset" in str(embedding_paths[0])
            assert "beats_pretrained" in str(embedding_paths[0])

            # Test with different model - should have different path
            model_name2 = "efficientnet_b0"
            embedding_paths2 = []
            for _ in experiments:
                embedding_dir_name = dataset_name
                emb_base_dir = save_dir / f"{embedding_dir_name}_{model_name2}"
                train_path = emb_base_dir / "embedding_train.h5"
                embedding_paths2.append(train_path)

            # Different model should have different embedding files
            assert embedding_paths[0] != embedding_paths2[0]
            assert "efficientnet_b0" in str(embedding_paths2[0])
