"""
Utilities for experiment tracking using pandas DataFrames.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from representation_learning.configs import RunConfig


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
        "timestamp": datetime.now().isoformat(),
        "checkpoint_name": checkpoint_name,
        "is_best": is_best,
        "is_final": is_final,
        "config": json.dumps(config_dict),  # Store config as JSON string
    }

    # Add metrics if provided
    if metrics:
        metadata.update(metrics)

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
    all_possible_train_metrics = {"loss", "acc"}
    all_possible_val_metrics = {"loss", "acc"}
    all_possible_test_metrics = {
        "accuracy",
        "balanced_accuracy",
        "multiclass_f1",
        "map",
        "roc_auc",
    }
    all_possible_retrieval_metrics = {"retrieval_roc_auc", "retrieval_precision_at_1"}

    # Create metadata entry with all possible metrics, using None for missing ones
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "experiment_name": experiment_name,
        "checkpoint_name": checkpoint_name,
    }

    # Add train metrics with None for missing ones
    for metric in all_possible_train_metrics:
        metadata[metric] = train_metrics.get(metric, None)

    # Add validation metrics with None for missing ones
    for metric in all_possible_val_metrics:
        metadata[f"val_{metric}"] = val_metrics.get(metric, None)

    # Add test metrics with None for missing ones
    for metric in all_possible_test_metrics:
        metadata[f"test_{metric}"] = probe_test_metrics.get(metric, None)

    # Add retrieval metrics with None for missing ones
    for metric in all_possible_retrieval_metrics:
        # Remove the "retrieval_" prefix if it's already there to avoid double-prefixing
        metric_name = metric.replace("retrieval_", "")
        metadata[f"retrieval_{metric_name}"] = retrieval_metrics.get(metric, None)

    metadata["eval_config"] = json.dumps(
        eval_config
    )  # Store eval config as JSON string

    # Add training metadata if available
    if training_metadata is not None and not training_metadata.empty:
        # Get the most recent training entry and convert to dict
        latest_training = training_metadata.iloc[-1].to_dict()

        # Store training metadata as JSON string
        metadata["training_params"] = json.dumps(latest_training)

    # Add run config if available (for pre-trained models)
    if run_config is not None:
        # Store run config as JSON string
        metadata["run_config_params"] = json.dumps(run_config)

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


class ExperimentLogger:
    """Logger for tracking experiment metrics and metadata."""

    def __init__(self, output_dir: Path, config: RunConfig) -> None:
        """Initialize the experiment logger.

        Parameters
        ----------
        output_dir : Path
            Directory to save experiment logs
        config : RunConfig
            Training configuration
        """
        self.output_dir = output_dir
        self.config = config
        self.metadata_dir = output_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: RunConfig) -> ExperimentLogger:
        """Create an ExperimentLogger from a config.

        Parameters
        ----------
        config : RunConfig
            Training configuration

        Returns
        -------
        ExperimentLogger
            Initialized experiment logger
        """
        output_dir = Path(config.output_dir)
        return cls(output_dir=output_dir, config=config)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics for the current step.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary of metric names and values
        step : int
            Current training step
        """
        metrics_file = self.metadata_dir / "metrics.csv"

        # Add timestamp and step
        entry = {"timestamp": datetime.now().isoformat(), "step": step, **metrics}

        # Convert to DataFrame
        df = pd.DataFrame([entry])

        # Save to CSV
        if metrics_file.exists():
            df.to_csv(metrics_file, mode="a", header=False, index=False)
        else:
            df.to_csv(metrics_file, index=False)

    def log_checkpoint(
        self,
        checkpoint_name: str,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Log checkpoint metadata.

        Parameters
        ----------
        checkpoint_name : str
            Name of the checkpoint file
        metrics : Optional[Dict[str, float]], optional
            Current metrics, by default None
        is_best : bool, optional
            Whether this is the best checkpoint, by default False
        is_final : bool, optional
            Whether this is the final checkpoint, by default False
        """
        save_experiment_metadata(
            output_dir=self.output_dir,
            config=self.config,
            checkpoint_name=checkpoint_name,
            metrics=metrics,
            is_best=is_best,
            is_final=is_final,
        )

    def get_latest_metrics(self) -> pd.DataFrame:
        """Get the latest metrics from the log.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the latest metrics
        """
        metrics_file = self.metadata_dir / "metrics.csv"
        if not metrics_file.exists():
            return pd.DataFrame()

        df = pd.read_csv(metrics_file)
        return df.iloc[-1] if not df.empty else pd.DataFrame()
