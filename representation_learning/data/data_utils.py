"""Data utility functions for dataset processing and balancing.

This module provides utility functions for data preprocessing, balancing,
and manipulation operations commonly used in representation learning tasks.
"""

from __future__ import annotations

from typing import List

import pandas as pd


def balance_by_attribute(
    dataset: pd.DataFrame,
    attribute: str,
    strategy: str = "undersample",
    target_count: int = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balance a pandas DataFrame by the specified attribute.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The input dataset to be balanced
    attribute : str
        The column name to balance by
    strategy : str, optional (default='undersample')
        Strategy for balancing:
        - 'undersample': Reduce all classes to the size of the smallest class
        - 'oversample': Increase all classes to the size of the largest class
        - 'target': Set all classes to a specific count defined by target_count
    target_count : int, optional (default=None)
        Target sample count per class when using 'target' strategy
    random_state : int, optional (default=42)
        Random seed for reproducibility

    Returns:
    --------
    pandas.DataFrame
        Balanced dataset

    Raises:
        ValueError: If attribute is not found, or if strategy/target_count is invalid.
    """
    if attribute not in dataset.columns:
        raise ValueError(f"Attribute '{attribute}' not found in dataset columns")

    # Get value counts
    value_counts = dataset[attribute].value_counts()
    min_count = value_counts.min()
    max_count = value_counts.max()

    # Determine target counts based on strategy
    if strategy == "undersample":
        target_counts = {val: min_count for val in value_counts.index}
    elif strategy == "oversample":
        target_counts = {val: max_count for val in value_counts.index}
    elif strategy == "target":
        if target_count is None:
            raise ValueError(
                "target_count must be specified when using 'target' strategy"
            )
        target_counts = {val: target_count for val in value_counts.index}
    else:
        raise ValueError(
            "Strategy must be one of: 'undersample', 'oversample', 'target'"
        )

    # Create empty DataFrame to hold balanced data
    balanced_data = pd.DataFrame(columns=dataset.columns)

    # Balance each class
    for val, count in target_counts.items():
        class_data = dataset[dataset[attribute] == val]

        if len(class_data) > count:
            # Undersample
            balanced_class = class_data.sample(n=count, random_state=random_state)
        elif len(class_data) < count:
            # Oversample with replacement
            balanced_class = class_data.sample(
                n=count, replace=True, random_state=random_state
            )
        else:
            # Already at target count
            balanced_class = class_data

        balanced_data = pd.concat([balanced_data, balanced_class], ignore_index=True)

    # Shuffle the final dataset
    return balanced_data.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )


def resample() -> None:
    """Placeholder for resample function."""
    pass


def combine_text_labels(label1: str, label2: str, *, delimiter: str = ", ") -> str:
    """Combine two text labels for CLIP mixup.

    Duplicates are collapsed so that "cat, cat" → "cat".
    The order is preserved as (label1, label2) unless they are identical.

    Returns
    -------
    str
        Combined label string with duplicates removed
    """

    if label1 == label2:
        return label1

    # Preserve insertion order but remove duplicates
    parts: List[str] = []
    for lbl in (label1, label2):
        if lbl not in parts:
            parts.append(lbl)
    return delimiter.join(parts)
