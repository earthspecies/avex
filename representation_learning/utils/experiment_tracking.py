"""
Utilities for experiment tracking using pandas DataFrames.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from esp_data.io import anypath, filesystem_from_path

from representation_learning.configs import RunConfig

# Global experiment directory for saving metadata
_GLOBAL_EXPERIMENT_DIR = "gs://representation-learning/experiment_results/"
_fs = filesystem_from_path(_GLOBAL_EXPERIMENT_DIR)


def _generate_run_id() -> str:
    return str(uuid.uuid5)


def save_experiment_metadata(
    output_dir: Path,
    config: RunConfig,
    checkpoint_name: str,
    metrics: Optional[Dict[str, float]] = None,
    is_best: bool = False,
    is_final: bool = False,
) -> None:
    """Save experiment metadata to a CSV file.

    Parameters
    ----------
    output_dir : Path
        Directory to save metadata
    config : RunConfig
        Training configuration
    checkpoint_name : str
        Name of the checkpoint file
    metrics : Optional[Dict[str, float]], optional
        Current metrics, by default None
    is_best : bool, optional
        Whether this is the best checkpoint, by default False
    is_final : bool, optional
        Whether this is the final checkpoint, by default False
    """
    # Create metadata directory if it doesn't exist
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Convert config to dict
    config_dict = config.model_dump(mode="json")

    # Create metadata entry
    metadata = {
        "end_timestamp": datetime.now().isoformat(),
        "checkpoint_name": checkpoint_name,
        "is_best": is_best,
        "is_final": is_final,
        "config": json.dumps(config_dict),  # Store config as JSON string
    }

    # Add metrics if provided
    if metrics:
        metadata.update(metrics)

    # Save metadata as in bucket
    if "run_name" not in config_dict:
        if "run_id" in config_dict:
            run_id = config_dict["run_id"]
        else:
            run_id = _generate_run_id()
        metadata["id"] = run_id
    else:
        metadata["id"] = config_dict.get("run_name", None) or config_dict.get(
            "run_id", _generate_run_id()
        )

    # Convert to DataFrame
    df = pd.DataFrame([metadata])

    # Save to CSV
    metadata_file = metadata_dir / "experiment_metadata.csv"
    if metadata_file.exists():
        # Append to existing file
        df.to_csv(metadata_file, mode="a", header=False, index=False)
    else:
        # Create new file
        df.to_csv(metadata_file, index=False)


def load_experiment_metadata(output_dir: Path) -> pd.DataFrame:
    """Load experiment metadata from CSV file.

    Parameters
    ----------
    output_dir : Path
        Directory containing metadata

    Returns
    -------
    pd.DataFrame
        DataFrame containing experiment metadata
    """
    metadata_file = output_dir / "metadata" / "experiment_metadata.csv"
    if not metadata_file.exists():
        return pd.DataFrame()

    # Read CSV
    df = pd.read_csv(metadata_file)

    # Convert boolean columns back to proper type
    boolean_columns = ["is_best", "is_final"]
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    return df


def save_evaluation_metadata(
    output_dir: Path,
    dataset_name: str,
    experiment_name: str,
    checkpoint_name: str,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    probe_test_metrics: Dict[str, float],
    retrieval_metrics: Dict[str, float],
    clustering_metrics: Dict[str, float],
    eval_config: Dict[str, Any],
    training_metadata: Optional[pd.DataFrame] = None,
    run_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save evaluation metadata to a CSV file.

    Parameters
    ----------
    output_dir : Path
        Directory to save metadata
    dataset_name : str
        Name of the dataset
    experiment_name : str
        Name of the experiment
    checkpoint_name : str
        Name of the checkpoint file
    train_metrics : Dict[str, float]
        Training metrics
    val_metrics : Dict[str, float]
        Validation metrics
    probe_test_metrics : Dict[str, float]
        Test metrics from linear probe
    retrieval_metrics : Dict[str, float]
        Retrieval metrics
    clustering_metrics : Dict[str, float]
        Clustering metrics
    eval_config : Dict[str, Any]
        Evaluation configuration
    training_metadata : Optional[pd.DataFrame]
        Training metadata from checkpoint if available
    run_config : Optional[Dict[str, Any]]
        Run configuration from experiment config if using pre-trained model
    """
    # Create metadata directory if it doesn't exist
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Define all possible metrics to ensure consistent columns
    all_possible_train_metrics = [
        "loss",
        "acc",
    ]
    all_possible_val_metrics = ["loss", "acc"]
    all_possible_test_metrics = [
        "accuracy",
        "balanced_accuracy",
        "multiclass_f1",
        "map",
        "roc_auc",
    ]
    all_possible_retrieval_metrics = [
        "retrieval_roc_auc",
        "retrieval_precision_at_1",
    ]
    all_possible_clustering_metrics = [
        "clustering_ari",
        "clustering_nmi",
        "clustering_v_measure",
        "clustering_silhouette",
        "clustering_best_k",
        "clustering_ari_best",
        "clustering_nmi_best",
        "clustering_v_measure_best",
        "clustering_silhouette_best",
    ]

    # Create metadata entry with all possible metrics, using None for missing ones
    metadata = {
        "end_timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "experiment_name": experiment_name,
        "checkpoint_name": checkpoint_name,
    }
    # Add train metrics with None for missing ones
    for metric in all_possible_train_metrics:
        metadata[f"train_{dataset_name}_{metric}"] = train_metrics.get(metric, None)

    # Add validation metrics with None for missing ones
    for metric in all_possible_val_metrics:
        metadata[f"val_{dataset_name}_{metric}"] = val_metrics.get(metric, None)

    # Add test metrics with None for missing ones
    for metric in all_possible_test_metrics:
        metadata[f"test_{dataset_name}_{metric}"] = probe_test_metrics.get(metric, None)

    # Add retrieval metrics with None for missing ones
    for metric in all_possible_retrieval_metrics:
        metadata[f"test_{dataset_name}_{metric}"] = retrieval_metrics.get(metric, None)

    # Add clustering metrics with None for missing ones
    for metric in all_possible_clustering_metrics:
        metadata[f"test_{dataset_name}_{metric}"] = clustering_metrics.get(metric, None)

    metadata["eval_config"] = json.dumps(
        eval_config
    )  # Store eval config as JSON string

    # Add training metadata if available
    latest_training = None
    if training_metadata is not None and not training_metadata.empty:
        # Get the most recent training entry and convert to dict
        latest_training = training_metadata.iloc[-1].to_dict()

        # Store training metadata as JSON string
        metadata["training_params"] = json.dumps(latest_training)

    # Add run config if available (for pre-trained models)
    if run_config is not None:
        # Store run config as JSON string
        metadata["run_config_params"] = json.dumps(run_config)

    # Save metadata as in bucket
    if run_config is not None:
        run_id = run_config.get("run_name", _generate_run_id())
    elif latest_training is not None:
        run_id = latest_training.get("run_name", _generate_run_id())
    else:
        run_id = _generate_run_id()

    output_json_path = anypath(_GLOBAL_EXPERIMENT_DIR) / (run_id + ".json")
    with _fs.open(output_json_path, "w") as f:
        json.dump(output_json_path, f)

    # Convert to DataFrame
    df = pd.DataFrame([metadata])

    # Save to CSV
    metadata_file = metadata_dir / "evaluation_metadata.csv"
    if metadata_file.exists():
        # Append to existing file
        df.to_csv(metadata_file, mode="a", header=False, index=False)
    else:
        # Create new file
        df.to_csv(metadata_file, index=False)


def load_evaluation_metadata(output_dir: Path) -> pd.DataFrame:
    """Load evaluation metadata from CSV file.

    Parameters
    ----------
    output_dir : Path
        Directory containing metadata

    Returns
    -------
    pd.DataFrame
        DataFrame containing evaluation metadata
    """
    metadata_file = output_dir / "metadata" / "evaluation_metadata.csv"
    if not metadata_file.exists():
        return pd.DataFrame()
    return pd.read_csv(metadata_file)


def create_experiment_summary_csvs(
    all_results: List[Any],  # ExperimentResult from run_evaluate.py
    eval_cfg: Any,  # EvaluateConfig  # noqa: ANN401
    save_dir: Path,
    config_file_path: str,
    benchmark_eval_cfg: Any,  # noqa: ANN401
    evaluation_sets: List[tuple],
    experiments: List[Any],
) -> None:
    """Create and save experiment summary DataFrames and CSV files.

    This function handles:
    1. Collecting all possible metrics from results and configs
    2. Creating full summary DataFrame with all metrics
    3. Creating simple summary DataFrame with key metrics only
    4. Saving to local summary CSV
    5. Appending to global results CSV (if configured)

    Parameters
    ----------
    all_results : List[ExperimentResult]
        List of experiment results
    eval_cfg : EvaluateConfig
        Evaluation configuration
    save_dir : Path
        Directory to save results
    config_file_path : str
        Path to the config file used
    benchmark_eval_cfg : BenchmarkEvaluationConfig
        Benchmark evaluation configuration
    evaluation_sets : List[tuple]
        List of (eval_set_name, eval_set_data_cfg) tuples
    experiments : List[ExperimentConfig]
        List of experiment configurations
    """
    import logging

    logger = logging.getLogger(__name__)

    # First, collect all possible metrics from all datasets to ensure consistent columns
    all_possible_metrics = set()
    all_possible_val_metrics = set()
    all_possible_test_metrics = set()
    all_possible_retrieval_metrics = set()
    all_possible_clustering_metrics = set()

    # Collect metrics from all results
    for r in all_results:
        all_possible_metrics.update(r.train_metrics.keys())
        all_possible_val_metrics.update(r.val_metrics.keys())
        all_possible_test_metrics.update(r.probe_test_metrics.keys())
        all_possible_retrieval_metrics.update(r.retrieval_metrics.keys())
        all_possible_clustering_metrics.update(r.clustering_metrics.keys())

    # Also collect metrics from dataset configurations
    for eval_set_name, _eval_set_data_cfg in evaluation_sets:
        eval_set = benchmark_eval_cfg.get_evaluation_set(eval_set_name)
        all_possible_test_metrics.update(eval_set.metrics)

    # Add standard retrieval metrics that are always computed when retrieval is enabled
    if "retrieval" in eval_cfg.eval_modes:
        all_possible_retrieval_metrics.update(
            ["retrieval_roc_auc", "retrieval_precision_at_1"]
        )

    # Add standard clustering metrics that are always computed when clustering
    # is enabled
    if "clustering" in eval_cfg.eval_modes:
        all_possible_clustering_metrics.update(
            [
                "clustering_ari",
                "clustering_nmi",
                "clustering_v_measure",
                "clustering_silhouette",
                "clustering_best_k",
                "clustering_ari_best",
                "clustering_nmi_best",
                "clustering_v_measure_best",
                "clustering_silhouette_best",
            ]
        )

    # Add standard training/validation metrics that are always computed
    all_possible_metrics.update(["loss", "acc"])
    all_possible_val_metrics.update(["loss", "acc"])

    summary_data = []
    for r in all_results:
        # Get training metadata if available
        training_metadata = pd.DataFrame()
        for exp_cfg in experiments:
            if exp_cfg.run_name == r.experiment_name:
                if exp_cfg.checkpoint_path:
                    checkpoint_dir = Path(exp_cfg.checkpoint_path).parent
                    training_metadata = load_experiment_metadata(checkpoint_dir)
                break

        # Create summary entry with all possible metrics, using None for missing ones
        summary_entry = {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": r.dataset_name,
            "experiment_name": r.experiment_name,
            "evaluation_dataset_name": r.evaluation_dataset_name,
        }

        # Add train metrics with None for missing ones
        for metric in all_possible_metrics:
            summary_entry[metric] = r.train_metrics.get(metric, None)

        # Add validation metrics with None for missing ones
        for metric in all_possible_val_metrics:
            summary_entry[f"val_{metric}"] = r.val_metrics.get(metric, None)

        # Add test metrics with None for missing ones
        for metric in all_possible_test_metrics:
            summary_entry[f"test_{metric}"] = r.probe_test_metrics.get(metric, None)

        # Add retrieval metrics with None for missing ones
        for metric in all_possible_retrieval_metrics:
            # Remove the "retrieval_" prefix if it's already there to avoid
            # double-prefixing
            metric_name = metric.replace("retrieval_", "")
            summary_entry[f"retrieval_{metric_name}"] = r.retrieval_metrics.get(
                metric, None
            )

        # Add clustering metrics with None for missing ones
        for metric in all_possible_clustering_metrics:
            # Remove the "clustering_" prefix if it's already there to avoid
            # double-prefixing
            metric_name = metric.replace("clustering_", "")
            summary_entry[f"clustering_{metric_name}"] = r.clustering_metrics.get(
                metric, None
            )

        # Add training metadata if available
        if not training_metadata.empty:
            # Get the most recent training entry
            latest_training = training_metadata.iloc[-1]
            for col in training_metadata.columns:
                if col not in summary_entry:
                    summary_entry[f"training_{col}"] = latest_training[col]

        summary_data.append(summary_entry)

    # Create and save DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df_path = save_dir / f"summary_{datetime.now()}.csv"
    summary_jsonl_path = save_dir / f"summary_{datetime.now()}.jsonl"
    # quoting=1 is QUOTE_ALL for proper CSV escaping
    summary_df.to_csv(summary_df_path, index=False, quoting=1, escapechar="\\")
    summary_df.to_json(summary_jsonl_path, orient="records", lines=True)
    logger.info("Saved summary DataFrame to %s", summary_df_path)
    logger.info("Saved summary JSONL to %s", summary_jsonl_path)

    # Append to global results CSV if specified
    if eval_cfg.results_csv_path:
        results_csv_path = Path(eval_cfg.results_csv_path).expanduser()

        # Add some additional metadata for the global CSV
        global_summary_data = []
        for entry in summary_data:
            global_entry = entry.copy()
            global_entry["config_file"] = config_file_path
            global_entry["save_dir"] = str(save_dir)
            global_summary_data.append(global_entry)

        global_summary_df = pd.DataFrame(global_summary_data)

        # Check if the file exists to determine if we need headers
        file_exists = results_csv_path.exists()

        # Create parent directory if it doesn't exist
        results_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to the file (or create it with headers if it doesn't exist)
        global_summary_df.to_csv(
            results_csv_path,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
            quoting=1,  # QUOTE_ALL to properly escape JSON and other special content
            escapechar="\\",
        )

        # Also save as JSONL
        results_jsonl_path = results_csv_path.with_suffix(".jsonl")

        # For JSONL, we always append (no headers needed)
        with open(results_jsonl_path, "a") as f:
            global_summary_df.to_json(f, orient="records", lines=True)

        logger.info("Appended results to global CSV: %s", results_csv_path)
        logger.info("Appended results to global JSONL: %s", results_jsonl_path)

    # Create simple CSV with just model name, date, dataset, and test metrics
    if eval_cfg.results_csv_path:
        # Derive simple CSV name from all_results CSV name
        results_path = Path(eval_cfg.results_csv_path).expanduser()
        simple_csv_path = results_path.parent / (
            results_path.stem + "_simple" + results_path.suffix
        )
    else:
        # Use a default name in the save_dir
        simple_csv_path = (
            save_dir / f"simple_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

    # Create simple summary data
    simple_summary_data = []
    for entry in summary_data:
        simple_entry = {
            "model_name": entry["experiment_name"],
            "date": entry["timestamp"],
            "dataset": entry.get("evaluation_dataset_name") or entry["dataset_name"],
        }

        # Add all test metrics (those that start with "test_")
        for key, value in entry.items():
            if key.startswith("test_") and value is not None:
                simple_entry[key] = value

        simple_summary_data.append(simple_entry)

    # Create and save simple DataFrame
    simple_summary_df = pd.DataFrame(simple_summary_data)

    # Check if the simple file exists to determine if we need headers
    simple_file_exists = simple_csv_path.exists()

    # Create parent directory if it doesn't exist
    simple_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to the simple file (or create it with headers if it doesn't exist)
    simple_summary_df.to_csv(
        simple_csv_path,
        mode="a" if simple_file_exists else "w",
        header=not simple_file_exists,
        index=False,
        quoting=1,  # QUOTE_ALL
        escapechar="\\",
    )

    # Also save as JSONL
    simple_jsonl_path = simple_csv_path.with_suffix(".jsonl")

    # For JSONL, we always append (no headers needed)
    with open(simple_jsonl_path, "a") as f:
        simple_summary_df.to_json(f, orient="records", lines=True)

    logger.info("Saved simple results CSV: %s", simple_csv_path)
    logger.info("Saved simple results JSONL: %s", simple_jsonl_path)


def parse_config_string(config_str: str) -> Dict[str, Any]:
    """Parse a config string back to a dictionary.

    Parameters
    ----------
    config_str : str
        JSON string representation of config

    Returns
    -------
    Dict[str, Any]
        Parsed config dictionary
    """
    try:
        return json.loads(config_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def parse_training_params_string(training_params_str: str) -> Dict[str, Any]:
    """Parse training parameters string back to a dictionary.

    Parameters
    ----------
    training_params_str : str
        JSON string representation of training parameters

    Returns
    -------
    Dict[str, Any]
        Parsed training parameters dictionary
    """
    try:
        return json.loads(training_params_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def parse_run_config_params_string(run_config_params_str: str) -> Dict[str, Any]:
    """Parse run config parameters string back to a dictionary.

    Parameters
    ----------
    run_config_params_str : str
        JSON string representation of run config parameters

    Returns
    -------
    Dict[str, Any]
        Parsed run config parameters dictionary
    """
    try:
        return json.loads(run_config_params_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def get_config_from_metadata(
    metadata_df: pd.DataFrame, config_column: str = "config"
) -> Optional[Dict[str, Any]]:
    """Extract config from metadata DataFrame.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame containing metadata
    config_column : str, optional
        Name of the config column, by default "config"

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed config dictionary if available, None otherwise
    """
    if config_column not in metadata_df.columns or metadata_df.empty:
        return None

    config_str = metadata_df[config_column].iloc[-1]
    if pd.isna(config_str):
        return None

    return parse_config_string(config_str)


def get_training_params_from_metadata(
    metadata_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Extract training parameters from metadata DataFrame.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame containing metadata

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed training parameters dictionary if available, None otherwise
    """
    if "training_params" not in metadata_df.columns or metadata_df.empty:
        return None

    training_params_str = metadata_df["training_params"].iloc[-1]
    if pd.isna(training_params_str):
        return None

    return parse_training_params_string(training_params_str)


def get_run_config_params_from_metadata(
    metadata_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Extract run config parameters from metadata DataFrame.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame containing metadata

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed run config parameters dictionary if available, None otherwise
    """
    if "run_config_params" not in metadata_df.columns or metadata_df.empty:
        return None

    run_config_params_str = metadata_df["run_config_params"].iloc[-1]
    if pd.isna(run_config_params_str):
        return None

    return parse_run_config_params_string(run_config_params_str)


def create_initial_experiment_metadata(
    output_dir: Path,
    config: RunConfig,
    checkpoint_name: str,
) -> pd.DataFrame:
    """Create initial experiment metadata for pre-trained models.

    Parameters
    ----------
    output_dir : Path
        Directory to save metadata
    config : RunConfig
        Training configuration
    checkpoint_name : str
        Name of the checkpoint file

    Returns
    -------
    pd.DataFrame
        DataFrame containing initial experiment metadata
    """
    # Create metadata entry with empty metrics
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_name": checkpoint_name,
        "is_best": True,
        "is_final": True,
        "config": json.dumps(
            config.model_dump(mode="json")
        ),  # Store config as JSON string
    }

    # Convert to DataFrame
    df = pd.DataFrame([metadata])

    # Save to CSV
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / "experiment_metadata.csv"
    df.to_csv(metadata_file, index=False)

    return df


def get_or_create_experiment_metadata(
    output_dir: Path,
    config: RunConfig,
    checkpoint_name: str,
) -> pd.DataFrame:
    """Get experiment metadata if it exists, otherwise create it.

    Parameters
    ----------
    output_dir : Path
        Directory containing or to save metadata
    config : RunConfig
        Training configuration
    checkpoint_name : str
        Name of the checkpoint file

    Returns
    -------
    pd.DataFrame
        DataFrame containing experiment metadata
    """
    metadata_file = output_dir / "metadata" / "experiment_metadata.csv"
    if metadata_file.exists():
        return pd.read_csv(metadata_file)
    else:
        return create_initial_experiment_metadata(output_dir, config, checkpoint_name)
