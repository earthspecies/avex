"""
Factory module for creating metric instances.
"""

from typing import Optional, Type, Union

from representation_learning.metrics.beans_metrics import (
    MeanAveragePrecision,
)
from representation_learning.metrics.sklearn_metrics import (
    ROCAUC,
    Accuracy,
    BalancedAccuracy,
    BinaryF1Score,
    MulticlassBinaryF1Score,
)


def get_metric_class(
    metric_name: str, num_classes: Optional[int] = None
) -> Union[Type, callable]:
    """Get the metric class based on the metric name.

    Args:
        metric_name: Name of the metric
        num_classes: Number of classes (required for some metrics)

    Returns:
        Metric class instance

    Raises:
        ValueError: If metric_name is not recognized
    """
    metric_map = {
        "accuracy": Accuracy,
        "balanced_accuracy": BalancedAccuracy,
        "binary_f1": BinaryF1Score,
        "multiclass_f1": lambda: MulticlassBinaryF1Score(num_classes),
        "map": MeanAveragePrecision,
        "mAP": MeanAveragePrecision,  # Support both map and mAP
        "roc_auc": ROCAUC,
    }

    if metric_name not in metric_map:
        raise ValueError(f"Unknown metric: {metric_name}")

    metric_class = metric_map[metric_name]
    if metric_name == "multiclass_f1":
        return metric_class()
    return metric_class()
