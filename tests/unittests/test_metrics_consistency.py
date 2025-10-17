#!/usr/bin/env python3
"""
Test script to verify that all possible metrics are included in the summary DataFrame,
even if they weren't computed for a particular dataset.
"""

from typing import Dict, List

import pandas as pd


# Simulate the results from different datasets with different metrics
def create_mock_results() -> List[Dict]:
    """Create mock experiment results with different metrics for different datasets.

    Returns:
        List[Dict]: List of mock experiment results with varying metrics
    """

    results = [
        {
            "dataset_name": "egyptian_fruit_bats",
            "experiment_name": "test_exp_1",
            "train_metrics": {"loss": 0.1, "acc": 0.95},
            "val_metrics": {"loss": 0.15, "acc": 0.92},
            "probe_test_metrics": {
                "accuracy": 0.93,
                "balanced_accuracy": 0.91,
            },
            "retrieval_metrics": {
                "retrieval_roc_auc": 0.88,
                "retrieval_precision_at_1": 0.85,
            },
        },
        {
            "dataset_name": "dogs",
            "experiment_name": "test_exp_1",
            "train_metrics": {"loss": 0.2, "acc": 0.88},
            "val_metrics": {"loss": 0.25, "acc": 0.85},
            "probe_test_metrics": {
                "accuracy": 0.87,
                "balanced_accuracy": 0.84,
            },
            "retrieval_metrics": {
                "retrieval_roc_auc": 0.82,
                "retrieval_precision_at_1": 0.79,
            },
        },
        {
            "dataset_name": "enabirds-detection",
            "experiment_name": "test_exp_1",
            "train_metrics": {"loss": 0.3, "acc": 0.75},
            "val_metrics": {"loss": 0.35, "acc": 0.72},
            "probe_test_metrics": {
                "multiclass_f1": 0.73
            },  # Only multiclass_f1, no accuracy
            "retrieval_metrics": {
                "retrieval_roc_auc": 0.76,
                "retrieval_precision_at_1": 0.73,
            },
        },
        {
            "dataset_name": "cbi",
            "experiment_name": "test_exp_1",
            "train_metrics": {"loss": 0.4, "acc": 0.65},
            "val_metrics": {"loss": 0.45, "acc": 0.62},
            "probe_test_metrics": {
                "accuracy": 0.64,
                "balanced_accuracy": 0.61,
            },  # No multiclass_f1
            "retrieval_metrics": {},  # No retrieval metrics
        },
    ]

    return results


def test_metrics_consistency() -> None:
    """Test that all possible metrics are included in the summary DataFrame."""

    # Get mock results
    all_results = create_mock_results()

    # Collect all possible metrics from all results (simulating our new logic)
    all_possible_metrics = set()
    all_possible_val_metrics = set()
    all_possible_test_metrics = set()
    all_possible_retrieval_metrics = set()

    # Collect metrics from all results
    for r in all_results:
        all_possible_metrics.update(r["train_metrics"].keys())
        all_possible_val_metrics.update(r["val_metrics"].keys())
        all_possible_test_metrics.update(r["probe_test_metrics"].keys())
        all_possible_retrieval_metrics.update(r["retrieval_metrics"].keys())

    # Add standard metrics that should always be present
    all_possible_metrics.update(["loss", "acc"])
    all_possible_val_metrics.update(["loss", "acc"])
    all_possible_retrieval_metrics.update(
        ["retrieval_roc_auc", "retrieval_precision_at_1"]
    )

    print("All possible metrics found:")
    print(f"  Train metrics: {sorted(all_possible_metrics)}")
    print(f"  Val metrics: {sorted(all_possible_val_metrics)}")
    print(f"  Test metrics: {sorted(all_possible_test_metrics)}")
    print(f"  Retrieval metrics: {sorted(all_possible_retrieval_metrics)}")

    # Create summary DataFrame with all possible metrics
    summary_data = []
    for r in all_results:
        # Create summary entry with all possible metrics, using None for missing ones
        summary_entry = {
            "dataset_name": r["dataset_name"],
            "experiment_name": r["experiment_name"],
        }

        # Add train metrics with None for missing ones
        for metric in all_possible_metrics:
            summary_entry[metric] = r["train_metrics"].get(metric, None)

        # Add validation metrics with None for missing ones
        for metric in all_possible_val_metrics:
            summary_entry[f"val_{metric}"] = r["val_metrics"].get(metric, None)

        # Add test metrics with None for missing ones
        for metric in all_possible_test_metrics:
            summary_entry[f"test_{metric}"] = r["probe_test_metrics"].get(metric, None)

        # Add retrieval metrics with None for missing ones
        for metric in all_possible_retrieval_metrics:
            # Remove the "retrieval_" prefix if it's already there to avoid
            # double-prefixing
            metric_name = metric.replace("retrieval_", "")
            summary_entry[f"retrieval_{metric_name}"] = r["retrieval_metrics"].get(
                metric, None
            )

        summary_data.append(summary_entry)

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)

    print("\nSummary DataFrame columns:")
    print(sorted(summary_df.columns.tolist()))

    print("\nSummary DataFrame:")
    print(summary_df.to_string())

    # Verify that all expected columns are present
    expected_columns = {
        "dataset_name",
        "experiment_name",
        "loss",
        "acc",
        "val_loss",
        "val_acc",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_multiclass_f1",
        "retrieval_roc_auc",
        "retrieval_precision_at_1",
    }

    actual_columns = set(summary_df.columns)
    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns

    print(f"\nMissing columns: {missing_columns}")
    print(f"Extra columns: {extra_columns}")

    # Verify that None values are present for missing metrics
    print("\nChecking for None values (missing metrics):")
    for col in summary_df.columns:
        if col.startswith(("test_", "retrieval_")):
            none_count = summary_df[col].isna().sum()
            if none_count > 0:
                print(f"  {col}: {none_count} None values")

    # Test passed if no missing expected columns
    assert len(missing_columns) == 0, f"Missing expected columns: {missing_columns}"
    print("\n✅ Test passed! All expected metrics are present in the DataFrame.")


if __name__ == "__main__":
    test_metrics_consistency()
