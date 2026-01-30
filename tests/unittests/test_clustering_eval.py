"""Tests for clustering evaluation functionality.

Copyright (c) 2024 Earth Species Project. All rights reserved.
"""

import pytest
import torch

from avex.evaluation.clustering import (
    eval_clustering,
    eval_clustering_multiple_k,
)

# Constants to avoid magic numbers
MIN_METRIC_THRESHOLD = 0.5
TOLERANCE = 1e-10
MIN_K = 2
MAX_K_SMALL = 5
MAX_K_MEDIUM = 6
MAX_K_LARGE = 10
MIN_SAMPLES_LARGE = 20


class TestEvalClustering:
    """Test cases for eval_clustering function."""

    def test_basic_clustering(self) -> None:
        """Test basic clustering with well-separated clusters."""
        # Create well-separated clusters
        cluster1 = torch.randn(20, 10) + torch.tensor(
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        cluster2 = torch.randn(20, 10) + torch.tensor(
            [-5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        cluster3 = torch.randn(20, 10) + torch.tensor(
            [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

        embeds = torch.cat([cluster1, cluster2, cluster3], dim=0)
        labels = torch.cat(
            [
                torch.zeros(20, dtype=torch.long),
                torch.ones(20, dtype=torch.long),
                torch.full((20,), 2, dtype=torch.long),
            ],
        )

        metrics = eval_clustering(embeds, labels)

        # Check that all expected metrics are present
        expected_metrics = {
            "clustering_ari",
            "clustering_nmi",
            "clustering_v_measure",
        }
        assert set(metrics.keys()) == expected_metrics

        # For well-separated clusters, metrics should be reasonably high
        assert metrics["clustering_ari"] > MIN_METRIC_THRESHOLD
        assert metrics["clustering_nmi"] > MIN_METRIC_THRESHOLD

    def test_custom_n_clusters(self) -> None:
        """Test clustering with custom number of clusters."""
        embeds = torch.randn(30, 5)
        labels = torch.randint(0, 3, (30,))

        # Test with different K than true number of classes
        metrics = eval_clustering(embeds, labels, n_clusters=5)

        # Should still return all metrics
        assert "clustering_ari" in metrics
        assert "clustering_nmi" in metrics
        assert "clustering_v_measure" in metrics

    def test_insufficient_clusters(self) -> None:
        """Test behavior with insufficient number of clusters."""
        embeds = torch.randn(10, 5)
        labels = torch.zeros(10, dtype=torch.long)  # All same label

        metrics = eval_clustering(embeds, labels)

        # Should return empty metrics due to only one class
        assert all(v == 0.0 for v in metrics.values())

    def test_more_clusters_than_samples(self) -> None:
        """Test behavior when requested clusters exceed samples."""
        embeds = torch.randn(5, 3)
        labels = torch.randint(0, 2, (5,))

        metrics = eval_clustering(embeds, labels, n_clusters=10)

        # Should handle gracefully and still return metrics
        assert "clustering_ari" in metrics

    def test_empty_inputs(self) -> None:
        """Test behavior with empty inputs."""
        embeds = torch.empty(0, 5)
        labels = torch.empty(0, dtype=torch.long)

        metrics = eval_clustering(embeds, labels)

        # Should return empty metrics
        assert all(v == 0.0 for v in metrics.values())

    def test_mismatched_shapes(self) -> None:
        """Test error handling for mismatched embedding and label shapes."""
        embeds = torch.randn(10, 5)
        labels = torch.randint(0, 3, (15,))  # Different length

        with pytest.raises(
            ValueError,
            match="Embeddings and labels must have same length",
        ):
            eval_clustering(embeds, labels)

    def test_multi_label_input(self) -> None:
        """Test handling of multi-label inputs."""
        embeds = torch.randn(20, 5)
        # Multi-hot labels (3 classes)
        labels = torch.zeros(20, 3)
        labels[0:7, 0] = 1  # First 7 samples have class 0
        labels[7:14, 1] = 1  # Next 7 samples have class 1
        labels[14:20, 2] = 1  # Last 6 samples have class 2

        metrics = eval_clustering(embeds, labels)

        # Should handle multi-label by taking argmax
        assert "clustering_ari" in metrics
        assert isinstance(metrics["clustering_ari"], float)

    def test_single_column_multi_label(self) -> None:
        """Test handling of single-column multi-label inputs."""
        embeds = torch.randn(20, 5)
        labels = torch.randint(0, 3, (20, 1))  # Single column

        metrics = eval_clustering(embeds, labels)

        # Should squeeze the labels and work normally
        assert "clustering_ari" in metrics

    def test_negative_labels(self) -> None:
        """Test handling of negative labels (e.g., unknown class)."""
        embeds = torch.randn(25, 5)
        labels = torch.tensor(
            [
                0,
                0,
                1,
                1,
                2,
                2,
                -1,
                -1,
                0,
                1,
                2,
                0,
                1,
                2,
                -1,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                -1,
            ],
        )

        metrics = eval_clustering(embeds, labels)

        # Should filter out negative labels when determining n_clusters
        assert "clustering_ari" in metrics

    def test_reproducibility(self) -> None:
        """Test that results are reproducible with fixed random state."""
        embeds = torch.randn(30, 5)
        labels = torch.randint(0, 3, (30,))

        metrics1 = eval_clustering(embeds, labels, random_state=42)
        metrics2 = eval_clustering(embeds, labels, random_state=42)

        # Results should be identical
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < TOLERANCE


class TestEvalClusteringMultipleK:
    """Test cases for eval_clustering_multiple_k function."""

    def test_basic_multiple_k(self) -> None:
        """Test multiple K clustering evaluation."""
        # Create data with clear 3 clusters
        cluster1 = torch.randn(15, 8) + torch.tensor(
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        cluster2 = torch.randn(15, 8) + torch.tensor(
            [-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        cluster3 = torch.randn(15, 8) + torch.tensor(
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

        embeds = torch.cat([cluster1, cluster2, cluster3], dim=0)
        labels = torch.cat(
            [
                torch.zeros(15, dtype=torch.long),
                torch.ones(15, dtype=torch.long),
                torch.full((15,), 2, dtype=torch.long),
            ],
        )

        metrics = eval_clustering_multiple_k(embeds, labels)

        # Check that all expected metrics are present
        expected_metrics = {
            "clustering_best_k",
            "clustering_ari_best",
            "clustering_nmi_best",
            "clustering_v_measure_best",
        }
        assert set(metrics.keys()) == expected_metrics

        # Best K should be close to true number of clusters (3)
        assert MIN_K <= metrics["clustering_best_k"] <= MAX_K_SMALL

    def test_custom_k_range(self) -> None:
        """Test multiple K clustering with custom range."""
        embeds = torch.randn(40, 6)
        labels = torch.randint(0, 4, (40,))

        metrics = eval_clustering_multiple_k(embeds, labels, k_range=(2, 6))

        assert "clustering_best_k" in metrics
        # Best K should be within the specified range
        assert MIN_K <= metrics["clustering_best_k"] <= MAX_K_MEDIUM

    def test_insufficient_samples_multiple_k(self) -> None:
        """Test multiple K with insufficient samples."""
        embeds = torch.randn(3, 4)
        labels = torch.randint(0, 2, (3,))

        metrics = eval_clustering_multiple_k(embeds, labels)

        # Should handle gracefully - with too few samples, may return empty metrics
        assert "clustering_best_k" in metrics
        # With very few samples, best_k might be 0 if no valid clustering is possible
        assert metrics["clustering_best_k"] >= 0

    def test_empty_inputs_multiple_k(self) -> None:
        """Test multiple K with empty inputs."""
        embeds = torch.empty(0, 5)
        labels = torch.empty(0, dtype=torch.long)

        metrics = eval_clustering_multiple_k(embeds, labels)

        # Should return empty metrics
        assert all(v == 0.0 for v in metrics.values())

    def test_auto_k_range_determination(self) -> None:
        """Test automatic K range determination."""
        embeds = torch.randn(50, 8)
        # 5 classes
        labels = torch.randint(0, 5, (50,))

        metrics = eval_clustering_multiple_k(embeds, labels)

        # Should automatically determine reasonable range around true K (5)
        assert "clustering_best_k" in metrics
        # Range should be reasonable around 5 classes
        assert MIN_K <= metrics["clustering_best_k"] <= MAX_K_LARGE

    def test_multi_label_multiple_k(self) -> None:
        """Test multiple K clustering with multi-label inputs."""
        embeds = torch.randn(30, 6)
        # Multi-hot labels
        labels = torch.zeros(30, 4)
        for i in range(30):
            labels[i, i % 4] = 1

        metrics = eval_clustering_multiple_k(embeds, labels)

        # Should handle multi-label appropriately
        assert "clustering_best_k" in metrics
        assert isinstance(metrics["clustering_best_k"], float)

    def test_reproducibility_multiple_k(self) -> None:
        """Test reproducibility of multiple K clustering."""
        embeds = torch.randn(35, 7)
        labels = torch.randint(0, 4, (35,))

        metrics1 = eval_clustering_multiple_k(embeds, labels, random_state=123)
        metrics2 = eval_clustering_multiple_k(embeds, labels, random_state=123)

        # Results should be identical
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < TOLERANCE

    def test_k_range_edge_cases(self) -> None:
        """Test edge cases for K range."""
        embeds = torch.randn(20, 5)
        labels = torch.randint(0, 3, (20,))

        # K range where max_k exceeds number of samples
        metrics = eval_clustering_multiple_k(embeds, labels, k_range=(2, 25))

        # Should handle gracefully and not try impossible K values
        assert "clustering_best_k" in metrics
        assert metrics["clustering_best_k"] < MIN_SAMPLES_LARGE  # Should be less than number of samples
