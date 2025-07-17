"""
Debug utilities for dataset inspection and logging.
"""

import logging

import pandas as pd
from esp_data import Dataset

logger = logging.getLogger(__name__)


def log_dataset_composition(ds: Dataset, dataset_type: str = "Training") -> None:
    """Log detailed composition information about a dataset.

    Parameters
    ----------
    ds : Dataset
        The dataset to analyze and log
    dataset_type : str, optional
        Type description for logging (e.g., "Training", "Validation"),
        by default "Training"
    """
    if not ds or ds._data is None:
        logger.info(f"{dataset_type} dataset is empty or None")
        return

    logger.info(f"=== {dataset_type.upper()} DATASET COMPOSITION DEBUG ===")
    # log first and last 2 captions
    if "caption" in ds._data.columns:
        logger.info(f"First 2 captions: {ds._data['caption'].head(2)}")
        logger.info(f"Last 2 captions: {ds._data['caption'].tail(2)}")

    # Check if we have dataset source info to break down by subdataset
    if hasattr(ds, "_source_info") or "dataset_name" in ds._data.columns:
        # Try to identify subdatasets
        if "dataset_name" in ds._data.columns:
            dataset_counts = ds._data["dataset_name"].value_counts()
            logger.info(f"{dataset_type} dataset composition by source:")
            for dataset_name, count in dataset_counts.items():
                logger.info(f"  {dataset_name}: {count} samples")

                # Show sample entries from each dataset
                sample_data = ds._data[ds._data["dataset_name"] == dataset_name].head(2)
                for idx, (_, row) in enumerate(sample_data.iterrows()):
                    # Show relevant fields for each dataset type
                    sample_info = []
                    if "canonical_name" in row and not pd.isna(row["canonical_name"]):
                        sample_info.append(f"canonical_name='{row['canonical_name']}'")
                    if "labels" in row and not pd.isna(row["labels"]):
                        sample_info.append(f"labels={row['labels']}")
                    if "caption" in row and not pd.isna(row["caption"]):
                        sample_info.append(f"caption='{str(row['caption'])[:50]}...'")
                    if "species_common" in row and not pd.isna(row["species_common"]):
                        sample_info.append(f"species_common='{row['species_common']}'")

                    sample_str = (
                        ", ".join(sample_info)
                        if sample_info
                        else "no relevant text fields"
                    )
                    logger.info(f"    Sample {idx + 1}: {sample_str}")
        else:
            logger.info(
                f"{dataset_type} dataset: {len(ds._data)} total samples "
                f"(no dataset_name field for breakdown)"
            )
    else:
        logger.info(f"{dataset_type} dataset: {len(ds._data)} total samples")

    # Show sample of columns available
    logger.info(f"Available columns: {list(ds._data.columns)}")
    logger.info(f"=== END {dataset_type.upper()} DATASET COMPOSITION DEBUG ===")
