"""
Temporary fixes for label handling issues in esp-data.
This file contains patches that should be removed once proper functionality
is added to esp-data.
"""

import ast
import logging

import pandas as pd
from esp_data import Dataset

logger = logging.getLogger(__name__)


def debug_labels_after_transformation(ds: Dataset) -> None:
    """
    Debug function to see what the labels look like after transformation.
    """
    if ds is None or ds._data is None:
        return

    logger.info("DEBUG: Checking labels after transformation...")

    # Check first 10 samples
    for idx in range(min(10, len(ds._data))):
        sample = ds._data.iloc[idx]
        label = sample.get("label", "NO_LABEL")
        unified_label = sample.get("unified_label", "NO_UNIFIED_LABEL")

        label_type = type(label)
        label_shape = (
            getattr(label, "shape", "no_shape")
            if hasattr(label, "shape")
            else "no_shape"
        )

        logger.info(
            f"Sample {idx}: label={repr(label)} (type: {label_type}, "
            f"shape: {label_shape})"
        )
        logger.info(f"  unified_label={repr(unified_label)}")

    # Check if there are any samples with different label tensor sizes
    if "label" in ds._data.columns:
        label_lengths = []
        label_values = []
        max_label_value = -1
        min_label_value = float("inf")

        for _idx, row in ds._data.iterrows():
            label = row.get("label")
            if hasattr(label, "__len__"):
                label_lengths.append(len(label))
                if hasattr(label, "__iter__"):
                    for val in label:
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            label_values.append(val)
                            max_label_value = max(max_label_value, val)
                            min_label_value = min(min_label_value, val)
            else:
                label_lengths.append(1)
                if isinstance(label, (int, float)) and not pd.isna(label):
                    label_values.append(label)
                    max_label_value = max(max_label_value, label)
                    min_label_value = min(min_label_value, label)

        unique_lengths = set(label_lengths)
        logger.info(f"DEBUG: Found label lengths: {unique_lengths}")
        logger.info(f"DEBUG: Label value range: {min_label_value} to {max_label_value}")
        logger.info(f"DEBUG: Total unique label values: {len(set(label_values))}")

        if len(unique_lengths) > 1:
            logger.warning(
                f"WARNING: Labels have inconsistent lengths: {unique_lengths}"
            )

        if max_label_value > 1000:  # This seems suspiciously high
            logger.warning(f"WARNING: Found very high label value: {max_label_value}")
            # Show some high label values for debugging
            high_values = [v for v in label_values if v > 1000][:10]
            logger.warning(f"Some high label values: {high_values}")


def fix_audioset_string_labels(ds: Dataset) -> None:
    """
    Temporary fix for AudioSet labels stored as string representations of lists.

    AudioSet CSV files contain labels like: "['Speech', 'Music']" (string)
    But they should be parsed as: ['Speech', 'Music'] (list)

    This function detects and fixes this issue by parsing string representations
    back into actual lists.

    Parameters
    ----------
    ds : Dataset
        The dataset to fix
    """
    if ds is None or ds._data is None:
        return

    # Check if we have a labels column that might need fixing
    if "labels" not in ds._data.columns:
        return

    # Sample first 10 rows to see what we're dealing with
    logger.info("DEBUG: Checking first 10 rows for label format...")
    sample_labels = []
    string_bracket_count = 0

    for idx, row in ds._data.head(10).iterrows():
        labels = row.get("labels")
        sample_labels.append(f"Row {idx}: {repr(labels)} (type: {type(labels)})")
        if (
            isinstance(labels, str)
            and labels.startswith("[")
            and labels.endswith("]")
            and not pd.isna(labels)
        ):
            string_bracket_count += 1

    for sample in sample_labels:
        logger.info(f"DEBUG: {sample}")

    logger.info(
        f"DEBUG: Found {string_bracket_count}/10 samples with string bracket format"
    )

    # Always try to fix, but log what we're doing
    logger.info("Attempting to fix AudioSet string-formatted labels...")

    # Fix the labels column
    fixed_labels = []
    string_bracket_fixed = 0
    string_single_fixed = 0
    list_kept = 0
    other_converted = 0

    for _idx, row in ds._data.iterrows():
        labels = row.get("labels")

        # Handle the case where labels might be NaN (float) or a list
        try:
            is_nan = pd.isna(labels)
            # Handle case where is_nan might be an array
            if hasattr(is_nan, "__len__") and not isinstance(is_nan, str):
                # If it's an array, check if all values are NaN
                is_nan = is_nan.all() if len(is_nan) > 0 else True
        except (ValueError, TypeError):
            # If pd.isna fails (e.g., on lists), assume it's not NaN
            is_nan = False

        if is_nan:
            fixed_labels.append(None)
        elif (
            isinstance(labels, str) and labels.startswith("[") and labels.endswith("]")
        ):
            try:
                parsed_labels = ast.literal_eval(labels)
                if isinstance(parsed_labels, list):
                    fixed_labels.append(parsed_labels)
                    string_bracket_fixed += 1
                else:
                    fixed_labels.append([str(parsed_labels)])
                    other_converted += 1
            except (ValueError, SyntaxError):
                fixed_labels.append([labels])
                other_converted += 1
        elif isinstance(labels, str):
            fixed_labels.append([labels])
            string_single_fixed += 1
        elif isinstance(labels, list):
            fixed_labels.append(labels)
            list_kept += 1
        else:
            fixed_labels.append([str(labels)])
            other_converted += 1

    ds._data["labels"] = fixed_labels
    logger.info(
        f"Fixed {string_bracket_fixed} bracket strings parsed, "
        f"{string_single_fixed} single strings, {list_kept} lists kept, "
        f"{other_converted} others converted"
    )

    # Sample the unique labels after fixing
    all_labels = set()
    for label_list in fixed_labels:
        if isinstance(label_list, list):
            for label in label_list:
                if label:
                    all_labels.add(str(label))

    logger.info(f"DEBUG: After fixing, found {len(all_labels)} unique labels total")
    sample_unique = sorted(list(all_labels))[:20]
    logger.info(f"DEBUG: Sample unique labels after fixing: {sample_unique}")
