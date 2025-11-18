"""Visualization and analysis of Excel data for representation learning experiments.

This module provides functions to load, process, and visualize results from
Excel files containing benchmark evaluation metrics.
"""

from pathlib import Path

import pandas as pd


def load_excel(file_path: str) -> pd.DataFrame:
    """Load an Excel file with multi-row headers and create proper column names.

    The Excel file structure:
    - Row 0: Description/metadata
    - Row 1: Benchmark groups (BEANS Classification, BirdSet, etc.)
    - Row 2: Dataset names (Watkins, POW, etc.)
    - Row 3: Metric names (Probe, R-auc, C-nmi, etc.)
    - Row 4+: Model data

    Creates column names in format: "dataset_benchmark_metric"

    Args:
        file_path: Path to the Excel file (supports ~ expansion).

    Returns:
        DataFrame with flattened column names and model data.

    Raises:
        FileNotFoundError: If the Excel file does not exist.
    """
    path = Path(file_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    # Read without headers first
    df_raw = pd.read_excel(path, header=None)

    # Extract the three header rows
    benchmarks_row = df_raw.iloc[1]  # Row 1: benchmark groups
    datasets_row = df_raw.iloc[2]  # Row 2: dataset names
    metrics_row = df_raw.iloc[3]  # Row 3: metric names

    # Build proper column names
    column_names = []
    current_benchmark = None
    current_dataset = None

    # First pass: identify all unique dataset names to avoid splitting them
    dataset_names = {}
    temp_benchmark = None
    for i in range(len(df_raw.columns)):
        benchmark_val = benchmarks_row[i]
        if pd.notna(benchmark_val) and benchmark_val and str(benchmark_val).strip():
            if "Colors:" not in str(benchmark_val) and "Unnamed" not in str(
                benchmark_val
            ):
                temp_benchmark = str(benchmark_val).strip()

        dataset_val = datasets_row[i]
        if pd.notna(dataset_val) and dataset_val and str(dataset_val).strip():
            # Store the full dataset name with its benchmark
            dataset_names[i] = (
                str(dataset_val).strip().replace(" ", "_"),
                temp_benchmark,
            )

    # Second pass: build column names using full dataset names
    for i in range(len(df_raw.columns)):
        # Track benchmark changes
        benchmark_val = benchmarks_row[i]
        if pd.notna(benchmark_val) and benchmark_val and str(benchmark_val).strip():
            if "Colors:" not in str(benchmark_val) and "Unnamed" not in str(
                benchmark_val
            ):
                current_benchmark = str(benchmark_val).strip()
                # Map "Repertoire" to "Vocal Repertoire" for consistency
                if current_benchmark == "Repertoire":
                    current_benchmark = "Vocal Repertoire"

        # Track dataset changes - use the full dataset name from first pass
        if i in dataset_names:
            current_dataset = dataset_names[i][0]

        # Get metric
        metric_val = metrics_row[i]

        # Build column name
        if i == 0:
            # First column is usually metadata
            column_names.append("metadata")
        elif i == 1:
            # Second column is usually model names
            column_names.append("model")
        elif current_dataset and pd.notna(metric_val):
            # Create simple name: dataset_metric
            # Each dataset belongs to only one benchmark, so benchmark name not needed
            metric_str = str(metric_val).strip()
            col_name = f"{current_dataset}_{metric_str}"
            column_names.append(col_name)
        else:
            # Fallback for any other columns
            column_names.append(f"Unnamed_{i}")

    # Create DataFrame with proper columns, starting from row 4 (data rows)
    df_data = df_raw.iloc[4:].copy()
    df_data.columns = column_names

    # Drop metadata column if it exists
    if "metadata" in df_data.columns:
        df_data = df_data.drop(columns=["metadata"])

    # Convert numeric columns to float
    for col in df_data.columns:
        if col.startswith("Unnamed_"):
            continue
        if col == "model":
            continue
        try:
            df_data[col] = pd.to_numeric(df_data[col], errors="coerce")
        except Exception:
            pass

    # Set model names as index (required by the scripts)
    if "model" in df_data.columns:
        # Drop rows where model name is empty/NaN
        df_data = df_data[df_data["model"].notna()]
        df_data = df_data[df_data["model"] != ""]
        df_data = df_data.set_index("model")

    return df_data


def main() -> None:
    """Main function for processing Excel data with metric definitions."""
    # Define metrics for different benchmarks
    metrics = {
        "BEANS Classification": ["Probe", "R-auc", "C-nmi"],
        "BEANS Detection": ["Probe", "R-auc"],
        "BirdSet": ["Probe", "R-auc"],
        "Individual ID": ["Probe", "R-auc"],
        "Vocal Repertoire": ["R-auc", "C-nmi"],
    }

    # This metrics variable will be extracted by generate_enhanced_aggregate_table.py
    return metrics


if __name__ == "__main__":
    main()
