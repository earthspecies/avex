"""Integration tests for clustering evaluation in the full evaluation pipeline.

Copyright (c) 2024 Earth Species Project. All rights reserved.
"""

import json
from dataclasses import dataclass

import pytest
import torch

from avex.evaluation.clustering import (
    _get_empty_clustering_metrics,
    eval_clustering,
)

# Constants to avoid magic numbers
TOLERANCE = 1e-12
MIN_METRIC_THRESHOLD = 0.5
# SILHOUETTE_THRESHOLD removed - silhouette metric no longer computed
EXPECTED_ARI_VALUE = 0.75


@dataclass
class MockEvalConfig:
    """Mock evaluation configuration for testing."""

    eval_modes: list[str]


class TestClusteringIntegration:
    """Integration tests for clustering in the evaluation pipeline."""

    def test_clustering_integration_with_mock_data(self) -> None:
        """Test that clustering integrates properly with the evaluation pipeline."""
        # Create mock embeddings and labels
        n_samples = 100
        embed_dim = 64
        n_classes = 5

        # Create well-separated embeddings for each class
        embeds = []
        labels = []

        for class_id in range(n_classes):
            # Create cluster centered at different points with better separation
            cluster_center = torch.zeros(embed_dim)
            cluster_center[class_id * 3] = 10.0  # More separation between clusters

            class_embeds = torch.randn(n_samples // n_classes, embed_dim) * 0.5 + cluster_center  # Tighter clusters
            class_labels = torch.full(
                (n_samples // n_classes,),
                class_id,
                dtype=torch.long,
            )

            embeds.append(class_embeds)
            labels.append(class_labels)

        all_embeds = torch.cat(embeds, dim=0)
        all_labels = torch.cat(labels, dim=0)

        # Test clustering directly
        metrics = eval_clustering(all_embeds, all_labels)

        # Verify that all expected metrics are computed
        expected_metrics = {
            "clustering_ari",
            "clustering_nmi",
            "clustering_v_measure",
        }
        assert set(metrics.keys()) == expected_metrics

        # For well-separated clusters, metrics should be reasonably high
        assert metrics["clustering_ari"] > MIN_METRIC_THRESHOLD
        assert metrics["clustering_nmi"] > MIN_METRIC_THRESHOLD
        # Silhouette metric was removed for speed

    @pytest.mark.parametrize(
        "eval_modes",
        [
            ["clustering"],
            ["linear_probe", "clustering"],
            ["retrieval", "clustering"],
            ["linear_probe", "retrieval", "clustering"],
        ],
    )
    def test_eval_modes_with_clustering(self, eval_modes: list[str]) -> None:
        """Test that clustering works with various evaluation mode combinations."""
        # This test verifies that the clustering evaluation mode is properly
        # recognized and doesn't interfere with other evaluation modes.

        # Mock the evaluation config
        eval_cfg = MockEvalConfig(eval_modes)

        # Test the logic from run_evaluate.py
        need_clustering = "clustering" in eval_cfg.eval_modes
        need_retrieval = "retrieval" in eval_cfg.eval_modes
        need_probe = "linear_probe" in eval_cfg.eval_modes

        assert need_clustering == ("clustering" in eval_modes)
        assert need_retrieval == ("retrieval" in eval_modes)
        assert need_probe == ("linear_probe" in eval_modes)

        # Test that embedding recomputation logic includes clustering
        need_recompute_embeddings_test = need_retrieval or need_probe or need_clustering
        expected_recompute = any(mode in eval_modes for mode in ["retrieval", "linear_probe", "clustering"])
        assert need_recompute_embeddings_test == expected_recompute

    def test_clustering_metrics_serialization(self) -> None:
        """Test that clustering metrics can be properly serialized for saving."""
        # Create simple test data
        embeds = torch.randn(30, 8)
        labels = torch.randint(0, 3, (30,))

        metrics = eval_clustering(embeds, labels)

        # Test that all metrics are JSON-serializable (floats)
        # This should not raise an exception
        json_str = json.dumps(metrics)
        deserialized = json.loads(json_str)

        # Check that deserialized metrics match original
        for key, value in metrics.items():
            assert isinstance(deserialized[key], float)
            assert abs(deserialized[key] - value) < TOLERANCE

    def test_clustering_with_experiment_result_dataclass(self) -> None:
        """Test that clustering metrics integrate with ExperimentResult dataclass."""

        # Create a minimal ExperimentResult dataclass just for testing
        @dataclass
        class ExperimentResult:
            dataset_name: str
            experiment_name: str
            evaluation_dataset_name: str
            train_metrics: dict[str, float]
            val_metrics: dict[str, float]
            probe_test_metrics: dict[str, float]
            retrieval_metrics: dict[str, float]
            clustering_metrics: dict[str, float]

        # Create mock metrics
        clustering_metrics = {
            "clustering_ari": EXPECTED_ARI_VALUE,
            "clustering_nmi": 0.82,
            # clustering_silhouette removed for speed
        }

        # Create ExperimentResult with clustering metrics
        result = ExperimentResult(
            dataset_name="test_dataset",
            experiment_name="test_experiment",
            evaluation_dataset_name="test_eval_set",
            train_metrics={"loss": 0.5, "acc": 0.8},
            val_metrics={"loss": 0.6, "acc": 0.75},
            probe_test_metrics={"accuracy": 0.7},
            retrieval_metrics={"retrieval_roc_auc": 0.85},
            clustering_metrics=clustering_metrics,
        )

        # Verify that clustering metrics are properly stored
        assert result.clustering_metrics == clustering_metrics
        assert result.clustering_metrics["clustering_ari"] == EXPECTED_ARI_VALUE

    def test_clustering_with_benchmark_config(self) -> None:
        """Test that clustering can be used with benchmark configuration format.

        This simulates how clustering would be used with the
        benchmark_id_repertoire.yml configuration format.
        """
        # Mock dataset configuration (similar to what's in benchmark_id_repertoire.yml)
        mock_dataset_config = {
            "name": "zebra_finch_bird_id",
            "dataset_name": "zebra_finch_julie_elie",
            "sample_rate": 16000,
            "metrics": [
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "clustering_ari",
                "clustering_nmi",
            ],
        }

        # Verify that clustering metrics can be included in dataset configs
        assert "clustering_ari" in mock_dataset_config["metrics"]
        assert "clustering_nmi" in mock_dataset_config["metrics"]

        # Test that the metrics are properly handled
        expected_clustering_metrics = ["clustering_ari", "clustering_nmi"]
        dataset_clustering_metrics = [m for m in mock_dataset_config["metrics"] if m.startswith("clustering_")]
        assert set(dataset_clustering_metrics) == set(expected_clustering_metrics)

    def test_empty_clustering_metrics_handling(self) -> None:
        """Test handling of empty clustering metrics."""
        empty_metrics = _get_empty_clustering_metrics()

        # Verify all metrics are 0.0
        expected_keys = {
            "clustering_ari",
            "clustering_nmi",
            "clustering_v_measure",
            # clustering_silhouette removed for speed
        }
        assert set(empty_metrics.keys()) == expected_keys
        assert all(v == 0.0 for v in empty_metrics.values())

    def test_clustering_with_different_sample_sizes(self) -> None:
        """Test clustering evaluation with various sample sizes."""
        embed_dim = 16

        # Test with different sample sizes
        sample_sizes = [10, 50, 100, 500]

        for n_samples in sample_sizes:
            n_classes = min(
                5,
                n_samples // 3,
            )  # Ensure reasonable number of samples per class

            # Create embeddings
            embeds = torch.randn(n_samples, embed_dim)
            labels = torch.randint(0, n_classes, (n_samples,))

            metrics = eval_clustering(embeds, labels)

            # Should always return the expected metrics
            assert "clustering_ari" in metrics
            # clustering_silhouette removed for speed
            assert isinstance(metrics["clustering_ari"], float)

    def test_clustering_deterministic_behavior(self) -> None:
        """Test that clustering evaluation produces deterministic results."""
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Create test data
        embeds = torch.randn(60, 12)
        labels = torch.randint(0, 4, (60,))

        # Run clustering evaluation multiple times
        metrics1 = eval_clustering(embeds, labels, random_state=123)
        metrics2 = eval_clustering(embeds, labels, random_state=123)

        # Results should be identical
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < TOLERANCE

    def test_clustering_with_id_classification_scenario(self) -> None:
        """Test clustering in the context of individual ID classification."""
        # This simulates the use case described in benchmark_id_repertoire.yml
        # where we have individual animal ID classification tasks.

        n_individuals = 10  # e.g., 10 different zebra finches
        samples_per_individual = 15
        embed_dim = 128

        # Create embeddings that simulate individual ID clustering
        embeds = []
        labels = []

        for individual_id in range(n_individuals):
            # Create embeddings for this individual (should cluster together)
            individual_center = torch.randn(embed_dim) * 2  # Random but fixed center for this individual
            individual_embeds = torch.randn(samples_per_individual, embed_dim) * 0.5 + individual_center
            individual_labels = torch.full(
                (samples_per_individual,),
                individual_id,
                dtype=torch.long,
            )

            embeds.append(individual_embeds)
            labels.append(individual_labels)

        all_embeds = torch.cat(embeds, dim=0)
        all_labels = torch.cat(labels, dim=0)

        metrics = eval_clustering(all_embeds, all_labels)

        # For individual ID tasks, we expect good clustering performance
        # if the embeddings capture individual identity well
        assert metrics["clustering_ari"] >= 0.0  # Should be non-negative
        assert metrics["clustering_nmi"] >= 0.0  # Should be non-negative

        # Metrics should be reasonable (not all zeros unless clustering failed)
        assert sum(metrics.values()) > 0
