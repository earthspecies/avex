"""Script to validate that datasets referenced in a
benchmark config exist and can be loaded."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml
from esp_data import DatasetConfig, dataset_from_config

from representation_learning.configs import BenchmarkEvaluationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def validate_datasets(config_path: Path) -> bool:
    """Validate that all datasets in a benchmark config can be loaded.

    Parameters
    ----------
    config_path : Path
        Path to the benchmark evaluation config YAML file.

    Returns
    -------
    bool
        True if all datasets are valid, False otherwise.
    """
    logger.info(f"Loading benchmark config from: {config_path}")
    with config_path.open() as f:
        config_dict = yaml.safe_load(f)
    benchmark_cfg = BenchmarkEvaluationConfig.model_validate(config_dict)

    evaluation_sets = benchmark_cfg.get_all_evaluation_sets()
    logger.info(f"Found {len(evaluation_sets)} evaluation sets")

    all_valid = True
    datasets_checked = set()

    for eval_set_name, dataset_collection_cfg in evaluation_sets:
        logger.info(f"\nValidating evaluation set: {eval_set_name}")

        # Check train datasets
        if dataset_collection_cfg.train_datasets:
            for ds_cfg in dataset_collection_cfg.train_datasets:
                dataset_key = (ds_cfg.dataset_name, ds_cfg.split, "train")
                if dataset_key not in datasets_checked:
                    if not _try_load_dataset(ds_cfg, "train", eval_set_name):
                        all_valid = False
                    datasets_checked.add(dataset_key)

        # Check validation datasets
        if dataset_collection_cfg.val_datasets:
            for ds_cfg in dataset_collection_cfg.val_datasets:
                dataset_key = (ds_cfg.dataset_name, ds_cfg.split, "val")
                if dataset_key not in datasets_checked:
                    if not _try_load_dataset(ds_cfg, "validation", eval_set_name):
                        all_valid = False
                    datasets_checked.add(dataset_key)

        # Check test datasets
        if dataset_collection_cfg.test_datasets:
            for ds_cfg in dataset_collection_cfg.test_datasets:
                dataset_key = (ds_cfg.dataset_name, ds_cfg.split, "test")
                if dataset_key not in datasets_checked:
                    if not _try_load_dataset(ds_cfg, "test", eval_set_name):
                        all_valid = False
                    datasets_checked.add(dataset_key)

    return all_valid


def _try_load_dataset(
    ds_cfg: DatasetConfig, split_name: str, eval_set_name: str
) -> bool:
    """Try to load a dataset and report success/failure.

    Parameters
    ----------
    ds_cfg : DatasetConfig
        Dataset configuration to load.
    split_name : str
        Name of the split (train/validation/test) for logging.
    eval_set_name : str
        Name of the evaluation set for logging.

    Returns
    -------
    bool
        True if dataset loaded successfully, False otherwise.
    """
    try:
        logger.info(
            f"  Checking {split_name} dataset: {ds_cfg.dataset_name} "
            f"(split: {ds_cfg.split})"
        )
        dataset, metadata = dataset_from_config(ds_cfg)
        num_samples = len(dataset) if dataset is not None else 0
        logger.info(
            f"  ✓ Successfully loaded {ds_cfg.dataset_name} "
            f"(split: {ds_cfg.split}): {num_samples} samples"
        )
        return True
    except Exception as e:
        logger.error(
            f"  ✗ Failed to load {ds_cfg.dataset_name} (split: {ds_cfg.split}): {e}"
        )
        return False


def main() -> None:
    """Main entry point for dataset validation."""
    if len(sys.argv) < 2:
        logger.error("Usage: python validate_datasets.py <config_path>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    is_valid = validate_datasets(config_path)

    if is_valid:
        logger.info("\n✓ All datasets are valid and can be loaded")
        sys.exit(0)
    else:
        logger.error("\n✗ Some datasets failed to load. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
