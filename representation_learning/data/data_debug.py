"""Utilities for debugging dataset pipeline issues."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetSampleSaver:
    """Utility class for saving dataset samples during pipeline execution."""

    def __init__(
        self, output_dir: str = ".", max_samples_per_dataset: int = 10
    ) -> None:
        """
        Initialize the sample saver.

        Args:
            output_dir: Directory to save debug files
            max_samples_per_dataset: Maximum number of samples to save per dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_samples_per_dataset = max_samples_per_dataset
        self.saved_samples = {
            "metadata": {
                "max_samples_per_dataset": max_samples_per_dataset,
                "saved_datasets": [],
            },
            "samples": {},
        }

    def save_dataset_samples(
        self, df: pd.DataFrame, dataset_name: str, stage: str = "processed"
    ) -> None:
        """
        Save a subset of samples from a dataset.

        Args:
            df: DataFrame containing the dataset
            dataset_name: Name identifier for the dataset
            stage: Processing stage (e.g., "raw", "processed", "concatenated")
        """
        try:
            # Create unique key for this dataset/stage combination
            key = f"{dataset_name}_{stage}"

            # Skip if already saved
            if key in self.saved_samples["samples"]:
                logger.debug(f"Samples for {key} already saved, skipping")
                return

            # Take sample
            sample_df = df.head(self.max_samples_per_dataset).copy()

            logger.info(f"Saving {len(sample_df)} samples from {key}")
            logger.info(f"Dataset columns: {list(sample_df.columns)}")

            # Convert to serializable format
            samples = []
            for idx, row in sample_df.iterrows():
                sample = {
                    "dataset": dataset_name,
                    "stage": stage,
                    "index": int(idx),
                    "data": {},
                }

                # Extract all columns, handling pandas types
                for col in sample_df.columns:
                    value = row[col]

                    # Handle pandas NA/None values safely
                    try:
                        is_na = pd.isna(value)
                        # Handle case where is_na might be an array
                        if hasattr(is_na, "__len__") and not isinstance(is_na, str):
                            # If it's an array, check if all values are NA
                            is_na = is_na.all() if len(is_na) > 0 else True
                    except (ValueError, TypeError):
                        # If pd.isna fails (e.g., on complex objects),
                        # assume it's not NA
                        is_na = False

                    if is_na:
                        sample["data"][col] = None
                    elif isinstance(value, (list, tuple)):
                        sample["data"][col] = list(value)
                    else:
                        sample["data"][col] = str(value) if value is not None else None

                samples.append(sample)

            # Store samples
            self.saved_samples["samples"][key] = samples
            self.saved_samples["metadata"]["saved_datasets"].append(key)

            logger.info(f"Successfully saved {len(samples)} samples for {key}")

        except Exception as e:
            logger.error(f"Failed to save samples for {dataset_name}_{stage}: {e}")

    def save_to_file(self, filename: str = "dataset_debug_samples.json") -> None:
        """Save all collected samples to a JSON file."""
        try:
            output_path = self.output_dir / filename

            # Add timestamp to metadata
            import pandas as pd

            self.saved_samples["metadata"]["timestamp"] = pd.Timestamp.now().isoformat()
            self.saved_samples["metadata"]["total_samples"] = sum(
                len(samples) for samples in self.saved_samples["samples"].values()
            )

            with open(output_path, "w") as f:
                json.dump(self.saved_samples, f, indent=2, default=str)

            logger.info(f"Debug samples saved to {output_path}")
            logger.info(
                f"Total samples saved: "
                f"{self.saved_samples['metadata']['total_samples']}"
            )

        except Exception as e:
            logger.error(f"Failed to save debug samples to file: {e}")


# Global instance for easy access
_global_saver: Optional[DatasetSampleSaver] = None


def init_debug_saver(
    output_dir: str = ".", max_samples_per_dataset: int = 10
) -> DatasetSampleSaver:
    """Initialize the global debug saver instance.

    Returns:
        DatasetSampleSaver: The initialized global saver instance.
    """
    global _global_saver
    _global_saver = DatasetSampleSaver(output_dir, max_samples_per_dataset)
    return _global_saver


def save_dataset_debug_samples(
    df: pd.DataFrame, dataset_name: str, stage: str = "processed"
) -> None:
    """
    Convenience function to save dataset samples using the global saver.

    This is the main function to call from the pipeline.

    Args:
        df: DataFrame containing the dataset
        dataset_name: Name identifier for the dataset
        stage: Processing stage (e.g., "raw", "processed", "concatenated")
    """
    global _global_saver

    if _global_saver is None:
        logger.debug("Debug saver not initialized, initializing with defaults")
        init_debug_saver()

    _global_saver.save_dataset_samples(df, dataset_name, stage)


def finalize_debug_samples(filename: str = "dataset_debug_samples.json") -> None:
    """Save all collected debug samples to file."""
    global _global_saver

    if _global_saver is None:
        logger.warning("Debug saver not initialized, nothing to save")
        return

    _global_saver.save_to_file(filename)


def get_debug_summary() -> Dict[str, Any]:
    """Get a summary of collected debug samples.

    Returns:
        Dict[str, Any]: Summary of collected debug samples and metadata.
    """
    global _global_saver

    if _global_saver is None:
        return {"error": "Debug saver not initialized"}

    return {
        "datasets_saved": _global_saver.saved_samples["metadata"]["saved_datasets"],
        "total_samples": sum(
            len(samples) for samples in _global_saver.saved_samples["samples"].values()
        ),
        "max_samples_per_dataset": _global_saver.max_samples_per_dataset,
    }


# Environment variable to enable/disable debug saving
DEBUG_SAVE_ENABLED = os.getenv("DATASET_DEBUG_SAVE", "false").lower() in (
    "true",
    "1",
    "yes",
)


def conditional_save_debug_samples(
    df: pd.DataFrame, dataset_name: str, stage: str = "processed"
) -> None:
    """
    Conditionally save debug samples based on environment variable.

    This function only saves samples if DATASET_DEBUG_SAVE environment variable is set.
    """
    if DEBUG_SAVE_ENABLED:
        save_dataset_debug_samples(df, dataset_name, stage)
