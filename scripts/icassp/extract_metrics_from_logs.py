#!/usr/bin/env python3
"""
Script to extract computed metrics from log files and generate CSV output.

This script parses log files from base model + probe experiments and extracts
metrics into a CSV format with datasets as columns and base models as rows.
"""

import argparse
import csv
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml


def format_number_to_human_readable(num_str: str) -> str:
    """Convert a number string to human-readable format (K, M, B).

    Args:
        num_str: Number as string (e.g., "90010417", "2.40M")

    Returns:
        Human-readable format (e.g., "90.01M", "2.40M")
    """
    if not num_str or num_str in ["", "nan", "NaN"]:
        return num_str

    # If already in human-readable format, return as is
    if any(suffix in num_str.upper() for suffix in ["K", "M", "B"]):
        return num_str

    try:
        num = float(num_str)

        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.2f}K"
        else:
            return str(int(num))
    except (ValueError, TypeError):
        return num_str


def parse_probe_configuration(line: str) -> Optional[Dict[str, Any]]:
    """Parse probe configuration from log line.

    Args:
        line: Log line containing probe configuration

    Returns:
        Dict containing probe configuration or None if not found
    """
    # Pattern: Probe configuration: type=weighted_attention, layers=['last_layer'],
    # aggregation=none, input_processing=sequence, training_mode=online
    pattern = (
        r"Probe configuration: type=(\w+), layers=\[([^\]]+)\], "
        r"aggregation=(\w+), input_processing=(\w+), training_mode=(\w+)"
    )
    match = re.search(pattern, line)

    if match:
        return {
            "probe_type": match.group(1),
            "layers": match.group(2).replace("'", "").replace('"', ""),
            "aggregation": match.group(3),
            "input_processing": match.group(4),
            "training_mode": match.group(5),
        }

    # Debug: Check if line contains "Probe configuration" but doesn't match
    if "Probe configuration" in line:
        print(f"DEBUG: Found 'Probe configuration' but pattern didn't match: {line}")

    return None


def parse_parameter_counts(line: str) -> Optional[Dict[str, int]]:
    """Parse parameter counts from log line.

    Args:
        line: Log line containing parameter information

    Returns:
        Dict containing parameter counts or None if not found
    """
    # Pattern for probe parameters: Probe parameters: 2.47M trainable / 2.47M total
    probe_pattern = r"Probe parameters: ([\d.]+[KMB]?) trainable / ([\d.]+[KMB]?) total"
    probe_match = re.search(probe_pattern, line)

    # Pattern for base model parameters: Base model parameters: 0 trainable /
    # 90010417 total
    base_pattern = r"Base model parameters: ([\d.]+[KMB]?) trainable / ([\d.]+[KMB]?) total"
    base_match = re.search(base_pattern, line)

    # Pattern for online training model: Online training model → 2.40M trainable /
    # 92.41M total
    online_pattern = r"Online training model → ([\d.]+[KMB]?) trainable / ([\d.]+[KMB]?) total"
    online_match = re.search(online_pattern, line)

    result = {}

    if probe_match:
        result["probe_trainable"] = probe_match.group(1)
        result["probe_total"] = probe_match.group(2)

    if base_match:
        result["base_trainable"] = base_match.group(1)
        # Only use base model parameters if we don't already have human-readable format
        if "base_total" not in result:
            result["base_total"] = base_match.group(2)

    if online_match:
        # For online training model, the total includes both probe and base model
        # Prioritize human-readable format over raw numbers
        result["base_total"] = online_match.group(2)  # Always use human-readable format
        if "probe_trainable" not in result:
            result["probe_trainable"] = online_match.group(1)
        if "probe_total" not in result:
            result["probe_total"] = online_match.group(1)  # Same as trainable for probe

    return result if result else None


def parse_layer_weights(log_lines: List[str], start_idx: int) -> Optional[str]:
    """Parse normalized layer weights from log lines.

    Args:
        log_lines: List of all log lines
        start_idx: Starting index to search from

    Returns:
        Comma-separated string of normalized weights or None if not found
    """
    # Look for "Learned Layer Weights:" section
    weights_section_start = None
    for i in range(start_idx, min(start_idx + 100, len(log_lines))):
        if "Learned Layer Weights:" in log_lines[i]:
            weights_section_start = i
            break

    if weights_section_start is None:
        return None

    # Extract normalized weights from the table
    weights = []
    for i in range(weights_section_start + 3, min(weights_section_start + 20, len(log_lines))):
        line = log_lines[i].strip()

        # Skip empty lines and header separators
        if not line or "----" in line:
            continue

        # Pattern: Layer_1         -0.4250      0.0590       5.90        %
        # or: all             -0.5796      0.0505       5.05        %
        # Split by whitespace and look for lines starting with "Layer_" or "all"
        parts = line.split()
        if len(parts) >= 3 and (parts[0].startswith("Layer_") or parts[0] == "all"):
            try:
                normalized_weight = float(parts[2])
                weights.append(f"{normalized_weight:.4f}")
            except (ValueError, IndexError):
                continue
        elif parts and not (parts[0].startswith("Layer_") or parts[0] == "all"):
            # If we hit a line that doesn't start with "Layer_" or "all",
            # we've probably reached the end
            break

    return ",".join(weights) if weights else None


def parse_layer_weights_backwards(log_lines: List[str], start_idx: int) -> Optional[str]:
    """Parse normalized layer weights from log lines.

    Searching backwards from start_idx.

    Args:
        log_lines: List of all log lines
        start_idx: Starting index to search backwards from

    Returns:
        Comma-separated string of normalized weights or None if not found
    """
    # Look for "Learned Layer Weights:" section by searching backwards
    weights_section_start = None
    for i in range(max(0, start_idx - 5000), start_idx):
        if "Learned Layer Weights:" in log_lines[i]:
            weights_section_start = i
            break

    if weights_section_start is None:
        return None

    # Extract normalized weights from the table
    weights = []
    for i in range(weights_section_start + 3, min(weights_section_start + 20, len(log_lines))):
        line = log_lines[i].strip()

        # Skip empty lines and header separators
        if not line or "----" in line:
            continue

        # Pattern: Layer_1         -0.4250      0.0590       5.90        %
        # or: all             -0.5796      0.0505       5.05        %
        # Split by whitespace and look for lines starting with "Layer_" or "all"
        parts = line.split()
        if len(parts) >= 3 and (parts[0].startswith("Layer_") or parts[0] == "all"):
            try:
                normalized_weight = float(parts[2])
                weights.append(f"{normalized_weight:.4f}")
            except (ValueError, IndexError):
                continue
        elif parts and not (parts[0].startswith("Layer_") or parts[0] == "all"):
            # If we hit a line that doesn't start with "Layer_" or "all",
            # we've probably reached the end
            break

    return ",".join(weights) if weights else None


def parse_metrics_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse metrics from log line.

    Args:
        line: Log line containing metrics

    Returns:
        Dict containing parsed metrics or None if not found
    """
    # Pattern: [dataset | benchmark | base_model] probe-test: {'mAP': 0.23582476377487183}  # noqa: E501
    # | retrieval: n/a | clustering: n/a
    # Note: Allow for multiple spaces before probe-test
    pattern = (
        r"\[([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^\]]+)\]\s+probe-test:\s*\{([^}]+)\}\s*"
        r"\|\s*retrieval:\s*([^|]+)\s*\|\s*clustering:\s*(.+)"
    )
    match = re.search(pattern, line)

    if not match:
        # Try alternative patterns
        # Pattern 2: Look for any line with 'probe-test' and metrics
        if "probe-test" in line and "{" in line and "}" in line:
            # Try to extract metrics from this line
            metrics_match = re.search(r"\{([^}]+)\}", line)
            if metrics_match:
                metrics_str = metrics_match.group(1)
                # Try to parse mAP or accuracy
                if "mAP" in metrics_str:
                    map_match = re.search(r"'mAP':\s*([0-9.]+)", metrics_str)
                    if map_match:
                        return {
                            "dataset": "unknown",
                            "benchmark": "unknown",
                            "base_model": "unknown",
                            "metrics": {"mAP": float(map_match.group(1))},
                            "retrieval": "n/a",
                            "clustering": "n/a",
                        }
                elif "accuracy" in metrics_str:
                    acc_match = re.search(r"'accuracy':\s*([0-9.]+)", metrics_str)
                    if acc_match:
                        return {
                            "dataset": "unknown",
                            "benchmark": "unknown",
                            "base_model": "unknown",
                            "metrics": {"accuracy": float(acc_match.group(1))},
                            "retrieval": "n/a",
                            "clustering": "n/a",
                        }
        return None

    dataset = match.group(1).strip()
    benchmark = match.group(2).strip()
    base_model = match.group(3).strip()
    metrics_str = match.group(4).strip()
    retrieval = match.group(5).strip()
    clustering = match.group(6).strip()

    # Parse metrics dictionary
    metrics = {}
    try:
        # Handle both single quotes and double quotes
        metrics_str = metrics_str.replace("'", '"')
        import json

        metrics = json.loads("{" + metrics_str + "}")
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract mAP or accuracy manually
        map_match = re.search(r"'mAP':\s*([\d.]+)", metrics_str)
        acc_match = re.search(r"'accuracy':\s*([\d.]+)", metrics_str)

        if map_match:
            metrics["mAP"] = float(map_match.group(1))
        if acc_match:
            metrics["accuracy"] = float(acc_match.group(1))

    return {
        "dataset": dataset,
        "benchmark": benchmark,
        "base_model": base_model,
        "metrics": metrics,
        "retrieval": retrieval,
        "clustering": clustering,
    }


def extract_metrics_from_log(log_file_path: str) -> List[Dict[str, Any]]:
    """Extract metrics from a single log file.

    Args:
        log_file_path: Path to the log file

    Returns:
        List of dictionaries containing extracted metrics and metadata
    """
    results = []

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading log file {log_file_path}: {e}")
        return results

    print(f"Processing {len(lines)} lines from log file")

    # Process lines
    i = 0
    probe_config_count = 0
    metrics_count = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for probe configuration
        probe_config = parse_probe_configuration(line)

        if probe_config:
            probe_config_count += 1

            # Look ahead for parameter counts and metrics
            probe_params = None
            base_params = None
            layer_weights = None
            metrics_data = None
            metrics_line_idx = None

            # Search more broadly for parameter counts (look ahead up to 200 lines)
            for j in range(i + 1, min(i + 200, len(lines))):
                next_line = lines[j].strip()

                # Check for parameter counts
                param_info = parse_parameter_counts(next_line)
                if param_info:
                    if "probe_trainable" in param_info:
                        probe_params = param_info
                    elif "base_trainable" in param_info:
                        base_params = param_info

            # Search more broadly for metrics (look ahead up to 200 lines)
            for j in range(i + 1, min(i + 300, len(lines))):
                next_line = lines[j].strip()

                # Check for metrics
                metrics_info = parse_metrics_line(next_line)
                if metrics_info:
                    metrics_data = metrics_info
                    metrics_line_idx = j
                    metrics_count += 1
                    break

            # If we found metrics, extract layer weights from after the metrics line
            if metrics_data and metrics_line_idx is not None:
                # Search forwards from the metrics line to find layer weights
                layer_weights = parse_layer_weights(lines, metrics_line_idx)

            # Combine all information
            if metrics_data:
                result = {
                    "probe_type": probe_config["probe_type"],
                    "layers": probe_config["layers"],
                    "aggregation": probe_config["aggregation"],
                    "input_processing": probe_config["input_processing"],
                    "training_mode": probe_config["training_mode"],
                    "dataset": metrics_data["dataset"],
                    "benchmark": metrics_data["benchmark"],
                    "base_model": metrics_data["base_model"],
                    "metrics": metrics_data["metrics"],
                    "retrieval": metrics_data["retrieval"],
                    "clustering": metrics_data["clustering"],
                    "probe_params": probe_params,
                    "base_params": base_params,
                    "layer_weights": layer_weights,
                }
                results.append(result)

        i += 1

    print(f"Found {probe_config_count} probe configurations and {metrics_count} metrics")
    return results


def extract_datasets_from_config(config_path: str) -> List[str]:
    """Extract dataset names from a benchmark config file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        List of dataset names
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    datasets = []
    for eval_set in config.get("evaluation_sets", []):
        datasets.append(eval_set["name"])

    return datasets


def get_expected_probe_types() -> List[str]:
    """Get the expected probe types for each experiment.

    Returns:
        List of expected probe type suffixes
    """
    return ["_linear_last", "_linear_all", "_attention_last", "_attention_all"]


def get_expected_probe_type_names() -> List[str]:
    """Get the expected probe type names.

    Returns:
        List of expected probe type names
    """
    return ["weighted_linear", "weighted_attention"]


def get_expected_layers() -> List[str]:
    """Get the expected layer configurations.

    Returns:
        List of expected layer configurations
    """
    return ["last_layer", "all"]


def read_extracted_csv(csv_path: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """Read the extracted CSV and get the datasets, probe types, and base models found.

    Args:
        csv_path: Path to the extracted metrics CSV

    Returns:
        Tuple of (datasets_found, probe_types_found, base_models_found)
    """
    datasets_found = set()
    probe_types_found = set()
    base_models_found = set()

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                datasets_found.add(row["dataset_name"])
                probe_types_found.add(row["probe_type"])
                base_models_found.add(row["base_model"])
    except FileNotFoundError:
        print(f"Warning: Could not read {csv_path}")

    return datasets_found, probe_types_found, base_models_found


def determine_benchmark_from_datasets(
    datasets_found: Set[str], birdset_datasets: List[str], beans_datasets: List[str]
) -> str:
    """Determine which benchmark the log file is testing based on found datasets.

    Args:
        datasets_found: Set of datasets found in the log
        birdset_datasets: List of expected birdset datasets
        beans_datasets: List of expected beans datasets

    Returns:
        'birdset', 'beans', or 'mixed' based on what datasets are found
    """
    birdset_found = any(dataset in datasets_found for dataset in birdset_datasets)
    beans_found = any(dataset in datasets_found for dataset in beans_datasets)

    if birdset_found and beans_found:
        return "mixed"
    elif birdset_found:
        return "birdset"
    elif beans_found:
        return "beans"
    else:
        return "unknown"


def generate_missing_combinations(
    birdset_datasets: List[str],
    beans_datasets: List[str],
    datasets_found: Set[str],
    probe_types_found: Set[str],
    base_models_found: Set[str],
) -> List[dict]:
    """Generate missing dataset and probe type combinations.

    Args:
        birdset_datasets: List of expected birdset datasets
        beans_datasets: List of expected beans datasets
        datasets_found: Set of datasets found in the log
        probe_types_found: Set of probe types found in the log
        base_models_found: Set of base models found in the log

    Returns:
        List of dictionaries containing missing combinations
    """
    missing_combinations = []

    # Determine which benchmark this log file is testing
    benchmark = determine_benchmark_from_datasets(datasets_found, birdset_datasets, beans_datasets)

    # Only check datasets from the relevant benchmark
    if benchmark == "birdset":
        relevant_datasets = birdset_datasets
        dataset_source = "birdset"
    elif benchmark == "beans":
        relevant_datasets = beans_datasets
        dataset_source = "beans"
    elif benchmark == "mixed":
        relevant_datasets = birdset_datasets + beans_datasets
        dataset_source = "mixed"
    else:
        print(f"Warning: Could not determine benchmark from datasets: {datasets_found}")
        return missing_combinations

    print(f"Detected benchmark: {benchmark}")
    print(f"Checking {len(relevant_datasets)} datasets from {dataset_source} benchmark")

    # Expected probe types and layers
    expected_probe_types = get_expected_probe_type_names()
    expected_layers = get_expected_layers()

    # Check only relevant datasets
    for dataset in relevant_datasets:
        for probe_type in expected_probe_types:
            for layer in expected_layers:
                # Create expected base model name
                # Map layer names to the correct format used in the logs
                layer_mapping = {"last_layer": "last", "all": "all"}
                mapped_layer = layer_mapping.get(layer, layer)
                base_model = f"sl_eat_all_ssl_all_{probe_type.replace('weighted_', '')}_{mapped_layer}"

                # Check if this combination is missing
                if (
                    dataset not in datasets_found
                    or probe_type not in probe_types_found
                    or base_model not in base_models_found
                ):
                    # Determine dataset source for this specific dataset
                    if dataset in birdset_datasets:
                        actual_dataset_source = "birdset"
                    elif dataset in beans_datasets:
                        actual_dataset_source = "beans"
                    else:
                        actual_dataset_source = "unknown"

                    missing_combinations.append(
                        {
                            "dataset_name": dataset,
                            "probe_type": probe_type,
                            "layers": layer,
                            "expected_base_model": base_model,
                            "dataset_source": actual_dataset_source,
                            "status": "missing",
                        }
                    )

    return missing_combinations


def split_csv_by_benchmark(input_csv_path: str, birdset_datasets: List[str], beans_datasets: List[str]) -> None:
    """Split the extracted metrics CSV into separate birdset and beans files.

    Args:
        input_csv_path: Path to the input CSV file
        birdset_datasets: List of birdset dataset names
        beans_datasets: List of beans dataset names
    """
    print("Splitting extracted metrics by benchmark...")

    # Read the input CSV
    try:
        with open(input_csv_path, "r") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
    except FileNotFoundError:
        print(f"Warning: Could not read {input_csv_path}")
        return

    # Split rows by benchmark
    birdset_rows = []
    beans_rows = []

    for row in all_rows:
        dataset_name = row.get("dataset_name", "")
        if dataset_name in birdset_datasets:
            birdset_rows.append(row)
        elif dataset_name in beans_datasets:
            beans_rows.append(row)

    # Write birdset CSV
    birdset_output = input_csv_path.replace(".csv", "_birdset.csv")
    with open(birdset_output, "w", newline="") as f:
        if birdset_rows:
            fieldnames = birdset_rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(birdset_rows)

    # Write beans CSV
    beans_output = input_csv_path.replace(".csv", "_beans.csv")
    with open(beans_output, "w", newline="") as f:
        if beans_rows:
            fieldnames = beans_rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(beans_rows)

    print(f"Split into {len(birdset_rows)} birdset rows -> {birdset_output}")
    print(f"Split into {len(beans_rows)} beans rows -> {beans_output}")


def generate_missing_datasets_csv(extracted_csv_path: str, output_csv_path: str) -> None:
    """Generate CSV files showing missing datasets and probe types for each benchmark.

    Args:
        extracted_csv_path: Path to the extracted metrics CSV
        output_csv_path: Path to the output missing datasets CSV
    """
    print("Generating missing datasets and probe types CSV...")

    # Extract datasets from configs
    print("Reading benchmark configurations...")
    birdset_datasets = extract_datasets_from_config("configs/data_configs/benchmark_birdset.yml")
    beans_datasets = extract_datasets_from_config("configs/data_configs/beans.yml")

    print(f"Found {len(birdset_datasets)} birdset datasets: {birdset_datasets}")
    print(f"Found {len(beans_datasets)} beans datasets: {beans_datasets}")

    # Read what was actually extracted
    print("Reading extracted metrics...")
    datasets_found, probe_types_found, base_models_found = read_extracted_csv(extracted_csv_path)

    print(f"Found {len(datasets_found)} datasets in log: {sorted(datasets_found)}")
    print(f"Found {len(probe_types_found)} probe types in log: {sorted(probe_types_found)}")
    print(f"Found {len(base_models_found)} base models in log: {sorted(base_models_found)}")

    # Determine which benchmark is being tested based on found datasets
    benchmark_detected = determine_benchmark_from_datasets(datasets_found, birdset_datasets, beans_datasets)
    print(f"Detected benchmark: {benchmark_detected}")

    # Generate missing combinations only for the detected benchmark
    print("Generating missing combinations for detected benchmark...")

    birdset_missing = []
    beans_missing = []

    if benchmark_detected == "birdset":
        birdset_missing = generate_missing_combinations_for_benchmark(
            birdset_datasets,
            datasets_found,
            probe_types_found,
            base_models_found,
            "birdset",
        )
        print(f"Generated {len(birdset_missing)} missing birdset combinations")
    elif benchmark_detected == "beans":
        beans_missing = generate_missing_combinations_for_benchmark(
            beans_datasets,
            datasets_found,
            probe_types_found,
            base_models_found,
            "beans",
        )
        print(f"Generated {len(beans_missing)} missing beans combinations")
    elif benchmark_detected == "mixed":
        # If mixed, generate for both benchmarks
        birdset_missing = generate_missing_combinations_for_benchmark(
            birdset_datasets,
            datasets_found,
            probe_types_found,
            base_models_found,
            "birdset",
        )
        beans_missing = generate_missing_combinations_for_benchmark(
            beans_datasets,
            datasets_found,
            probe_types_found,
            base_models_found,
            "beans",
        )
        print(
            f"Generated {len(birdset_missing)} missing birdset combinations and "
            f"{len(beans_missing)} missing beans combinations"
        )
    else:
        print("Warning: Could not determine benchmark from datasets")

    # Write combined missing CSV
    all_missing = birdset_missing + beans_missing
    print(f"Writing {len(all_missing)} missing combinations to {output_csv_path}...")
    with open(output_csv_path, "w", newline="") as f:
        if all_missing:
            fieldnames = all_missing[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_missing)
        else:
            # Write empty CSV with headers
            fieldnames = [
                "dataset_name",
                "probe_type",
                "layers",
                "expected_base_model",
                "dataset_source",
                "status",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Write separate files only for the detected benchmark
    if benchmark_detected in ["birdset", "mixed"]:
        birdset_output = output_csv_path.replace(".csv", "_birdset.csv")
        with open(birdset_output, "w", newline="") as f:
            if birdset_missing:
                fieldnames = birdset_missing[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(birdset_missing)
            else:
                fieldnames = [
                    "dataset_name",
                    "probe_type",
                    "layers",
                    "expected_base_model",
                    "dataset_source",
                    "status",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        print(f"Successfully wrote {len(birdset_missing)} birdset missing combinations to {birdset_output}")

    if benchmark_detected in ["beans", "mixed"]:
        beans_output = output_csv_path.replace(".csv", "_beans.csv")
        with open(beans_output, "w", newline="") as f:
            if beans_missing:
                fieldnames = beans_missing[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(beans_missing)
            else:
                fieldnames = [
                    "dataset_name",
                    "probe_type",
                    "layers",
                    "expected_base_model",
                    "dataset_source",
                    "status",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        print(f"Successfully wrote {len(beans_missing)} beans missing combinations to {beans_output}")

    # Summary
    print("\nSummary:")
    print(
        f"Expected datasets: {len(birdset_datasets + beans_datasets)} "
        f"(birdset: {len(birdset_datasets)}, beans: {len(beans_datasets)})"
    )
    print(f"Found datasets: {len(datasets_found)}")
    print(
        f"Missing datasets: {len(birdset_datasets + beans_datasets) - len(datasets_found)}"  # noqa: E501
    )
    print(f"Expected probe types: {len(get_expected_probe_type_names())}")
    print(f"Found probe types: {len(probe_types_found)}")
    print(
        f"Missing probe types: {len(get_expected_probe_type_names()) - len(probe_types_found)}"  # noqa: E501
    )


def generate_missing_combinations_for_benchmark(
    benchmark_datasets: List[str],
    datasets_found: Set[str],
    probe_types_found: Set[str],
    base_models_found: Set[str],
    benchmark_name: str,
) -> List[dict]:
    """Generate missing combinations for a specific benchmark.

    Args:
        benchmark_datasets: List of datasets for this benchmark
        datasets_found: Set of datasets found in the log
        probe_types_found: Set of probe types found in the log
        base_models_found: Set of base models found in the log
        benchmark_name: Name of the benchmark ('birdset' or 'beans')

    Returns:
        List of dictionaries containing missing combinations for this benchmark
    """
    missing_combinations = []

    # Expected probe types and layers
    expected_probe_types = get_expected_probe_type_names()
    expected_layers = get_expected_layers()

    # Check only datasets from this benchmark
    for dataset in benchmark_datasets:
        for probe_type in expected_probe_types:
            for layer in expected_layers:
                # Create expected base model name based on the actual base models found
                layer_mapping = {"last_layer": "last", "all": "all"}
                mapped_layer = layer_mapping.get(layer, layer)

                # Find the expected base model name by looking at what's actually in the log  # noqa: E501
                # We need to find a base model that matches the probe_type and layer pattern  # noqa: E501
                expected_base_model = None
                for base_model in base_models_found:
                    if probe_type.replace("weighted_", "") in base_model and mapped_layer in base_model:
                        expected_base_model = base_model
                        break

                # If no matching base model found, use a generic name
                if not expected_base_model:
                    if benchmark_name == "beans":
                        expected_base_model = (
                            f"efficientnet_animalspeak_audioset_{probe_type.replace('weighted_', '')}_{mapped_layer}"
                        )
                    else:  # birdset
                        expected_base_model = f"sl_eat_all_ssl_all_{probe_type.replace('weighted_', '')}_{mapped_layer}"

                # Check if this specific combination is missing
                # A combination is missing if the dataset is not found in the log
                # OR if the probe_type is not found in the log
                # OR if the expected base model is not found in the log
                is_missing = (
                    dataset not in datasets_found
                    or probe_type not in probe_types_found
                    or expected_base_model not in base_models_found
                )

                if is_missing:
                    missing_combinations.append(
                        {
                            "dataset_name": dataset,
                            "probe_type": probe_type,
                            "layers": layer,
                            "expected_base_model": expected_base_model,
                            "dataset_source": benchmark_name,
                            "status": "missing",
                        }
                    )

    return missing_combinations


def is_duplicate_entry(new_row: Dict[str, Any], existing_data: List[Dict[str, Any]]) -> bool:
    """Check if a new row already exists in the existing data.

    Args:
        new_row: New row to check
        existing_data: List of existing rows

    Returns:
        True if duplicate exists, False otherwise
    """
    for existing_row in existing_data:
        # Check if the key fields match (dataset, probe_type, layers, base_model)
        if (
            new_row["dataset_name"] == existing_row.get("dataset_name")
            and new_row["probe_type"] == existing_row.get("probe_type")
            and new_row["layers"] == existing_row.get("layers")
            and new_row["base_model"] == existing_row.get("base_model")
        ):
            return True
    return False


def create_or_append_csv(results: List[Dict[str, Any]], csv_file_path: str) -> None:
    """Create or append results to CSV file, avoiding duplicates.

    Args:
        results: List of extracted results
        csv_file_path: Path to the CSV file
    """
    if not results:
        print("No results to write to CSV")
        return

    # Check if CSV file exists
    file_exists = os.path.exists(csv_file_path)

    # Read existing data if file exists
    existing_data = []
    if file_exists:
        try:
            df_existing = pd.read_csv(csv_file_path)
            existing_data = df_existing.to_dict("records")
        except Exception as e:
            print(f"Warning: Could not read existing CSV file: {e}")

    # Prepare new data - each result becomes one row
    new_rows = []
    duplicate_count = 0

    for result in results:
        # Extract mAP or accuracy
        metric_value = ""
        if "mAP" in result["metrics"]:
            metric_value = result["metrics"]["mAP"]
        elif "accuracy" in result["metrics"]:
            metric_value = result["metrics"]["accuracy"]

        row = {
            "dataset_name": result["dataset"],
            "probe_type": result["probe_type"],
            "layers": result["layers"],
            "base_model": result["base_model"],
            "benchmark": result["benchmark"],
            "metric": metric_value,
        }

        # Add parameter information
        if result["probe_params"]:
            row["probe_trainable"] = result["probe_params"].get("probe_trainable", "")
            row["probe_total"] = result["probe_params"].get("probe_total", "")
        else:
            row["probe_trainable"] = ""
            row["probe_total"] = ""

        if result["base_params"]:
            row["base_trainable"] = result["base_params"].get("base_trainable", "")
            base_total_raw = result["base_params"].get("base_total", "")
            row["base_total"] = format_number_to_human_readable(base_total_raw)
        else:
            row["base_trainable"] = ""
            row["base_total"] = ""

        # Add layer weights
        row["layer_weights"] = result["layer_weights"] or ""

        # Check if this combination already exists
        if is_duplicate_entry(row, existing_data):
            duplicate_count += 1
            print(
                f"Skipping duplicate: {row['dataset_name']} | {row['probe_type']} | "
                f"{row['layers']} | {row['base_model']}"
            )
        else:
            new_rows.append(row)

    # Combine existing and new data
    all_data = existing_data + new_rows

    # Define column order
    column_order = [
        "dataset_name",
        "probe_type",
        "layers",
        "base_model",
        "benchmark",
        "probe_trainable",
        "probe_total",
        "base_trainable",
        "base_total",
        "layer_weights",
        "metric",
    ]

    # Write to CSV
    try:
        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=column_order)
            writer.writeheader()
            for row in all_data:
                # Fill missing columns with empty strings
                complete_row = {col: row.get(col, "") for col in column_order}
                writer.writerow(complete_row)

        print(f"Successfully wrote {len(new_rows)} new rows to {csv_file_path}")
        if duplicate_count > 0:
            print(f"Skipped {duplicate_count} duplicate entries")
        print(f"Total rows in CSV: {len(all_data)}")

        # Show unique datasets found
        datasets = set(row["dataset_name"] for row in all_data if row["dataset_name"])
        print(f"Datasets found: {sorted(datasets)}")

    except Exception as e:
        print(f"Error writing to CSV file: {e}")


def main() -> None:
    """Main function to extract metrics from log files."""
    parser = argparse.ArgumentParser(description="Extract metrics from log files and generate CSV output")
    parser.add_argument("log_file", type=str, help="Path to the log file to process")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="extracted_metrics.csv",
        help="Output CSV file path (default: extracted_metrics.csv)",
    )
    parser.add_argument(
        "--missing",
        "-m",
        type=str,
        help="Generate missing datasets CSV and save to specified path",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Check if log file exists
    if not os.path.exists(args.log_file):
        print(f"Error: Log file {args.log_file} not found")
        return

    print(f"Processing log file: {args.log_file}")

    # Extract metrics
    results = extract_metrics_from_log(args.log_file)

    if not results:
        print("No metrics found in the log file")
        return

    print(f"Found {len(results)} metric entries")

    if args.verbose:
        for i, result in enumerate(results):
            print(f"\nEntry {i + 1}:")
            print(f"  Dataset: {result['dataset']}")
            print(f"  Base Model: {result['base_model']}")
            print(f"  Probe Type: {result['probe_type']}")
            print(f"  Layers: {result['layers']}")
            print(f"  Metrics: {result['metrics']}")
            if result["layer_weights"]:
                print(f"  Layer Weights: {result['layer_weights']}")

    # Write to CSV
    create_or_append_csv(results, args.output)

    # Split extracted metrics by benchmark
    birdset_datasets = extract_datasets_from_config("configs/data_configs/benchmark_birdset.yml")
    beans_datasets = extract_datasets_from_config("configs/data_configs/beans.yml")
    split_csv_by_benchmark(args.output, birdset_datasets, beans_datasets)

    # Generate missing datasets CSV if requested
    if args.missing:
        generate_missing_datasets_csv(args.output, args.missing)


if __name__ == "__main__":
    main()
