"""
Utilities for experiment tracking using pandas DataFrames.
"""

from __future__ import annotations

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
        **config_dict,  # Include all config parameters
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

    # Create metadata entry
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "experiment_name": experiment_name,
        "checkpoint_name": checkpoint_name,
        **train_metrics,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in probe_test_metrics.items()},
        **{f"retrieval_{k}": v for k, v in retrieval_metrics.items()},
        **eval_config,  # Include all evaluation config parameters
    }

    # Add training metadata if available
    if training_metadata is not None and not training_metadata.empty:
        # Get the most recent training entry
        latest_training = training_metadata.iloc[-1]
        for col in training_metadata.columns:
            if col not in metadata:
                metadata[f"training_{col}"] = latest_training[col]

    # Add run config if available (for pre-trained models)
    if run_config is not None:
        for k, v in run_config.items():
            if k not in metadata:
                metadata[f"run_config_{k}"] = v

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
        **config.model_dump(mode="json"),  # Include all config parameters
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
