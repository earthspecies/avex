import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union

import gradio as gr
import pandas as pd
from gradio_leaderboard import Leaderboard


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Gradio leaderboard for experiment results from CSV file"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        help="Path to the CSV file containing experiment results",
    )
    parser.add_argument(
        "--training_params",
        type=str,
        help=(
            "Comma-delimited list of training parameter field names "
            "to extract from JSONL files"
        ),
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        help=(
            "Comma-delimited list of evaluation config field names "
            "to extract from JSONL files"
        ),
    )
    parser.add_argument(
        "--run_config_params",
        type=str,
        help=(
            "Comma-delimited list of run config parameter field names "
            "to extract from JSONL files"
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the Gradio server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the Gradio server to (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for the Gradio interface",
    )
    return parser.parse_args()


def load_data(
    csv_file_path: str,
    eval_config_fields: Optional[List[str]] = None,
    training_param_fields: Optional[List[str]] = None,
    run_config_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load data from CSV file with proper error handling and parameter extraction.

    Args:
        csv_file_path: Path to the CSV file to load
        eval_config_fields: List of field names to extract from eval_config
        training_param_fields: List of field names to extract from training_params
        run_config_fields: List of field names to extract from run_config_params

    Returns:
        pd.DataFrame: Loaded and processed DataFrame

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing or data is invalid
        ParserError: If the CSV file cannot be parsed after all strategies
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        # Define the base required columns
        base_required_columns = [
            "timestamp",
            "dataset_name",
            "experiment_name",
            "checkpoint_name",
            "retrieval_roc_auc",
            "retrieval_precision_at_1",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_roc_auc",
            "test_multiclass_f1",
            "test_map",
        ]

        # Add JSON configuration columns if parameter extraction is requested
        required_columns = base_required_columns.copy()
        if eval_config_fields:
            required_columns.append("eval_config")
        if training_param_fields:
            required_columns.append("training_params")
        if run_config_fields:
            required_columns.append("run_config_params")

        # Try different CSV parsing strategies with column selection
        df = None
        parsing_errors = []

        # Strategy 1: Try with default settings and column selection
        try:
            df = pd.read_csv(csv_file_path, usecols=required_columns)
            print("Successfully loaded CSV with default settings and column selection")
        except pd.errors.ParserError as e:
            parsing_errors.append(f"Default parsing failed: {e}")
        except ValueError as e:
            # Handle case where some columns don't exist
            parsing_errors.append(f"Column selection failed: {e}")

        # Strategy 2: Try with different quoting options and column selection
        if df is None:
            for quotechar in ['"', "'"]:
                try:
                    df = pd.read_csv(
                        csv_file_path,
                        quotechar=quotechar,
                        quoting=1,
                        usecols=required_columns,
                    )  # QUOTE_ALL
                    print(
                        f"Successfully loaded CSV with quotechar='{quotechar}' "
                        f"and QUOTE_ALL"
                    )
                    break
                except pd.errors.ParserError as e:
                    parsing_errors.append(f"Quotechar '{quotechar}' failed: {e}")
                except ValueError as e:
                    parsing_errors.append(
                        f"Column selection with quotechar '{quotechar}' failed: {e}"
                    )

        # Strategy 3: Try with error handling for bad lines and column selection
        if df is None:
            try:
                df = pd.read_csv(
                    csv_file_path,
                    on_bad_lines="skip",
                    engine="python",
                    usecols=required_columns,
                )
                print("Successfully loaded CSV with error handling (skipped bad lines)")
            except pd.errors.ParserError as e:
                parsing_errors.append(f"Error handling parsing failed: {e}")
            except ValueError as e:
                parsing_errors.append(
                    f"Column selection with error handling failed: {e}"
                )

        # Strategy 4: Try reading all columns first, then select
        if df is None:
            try:
                # Read all columns first
                all_df = pd.read_csv(
                    csv_file_path, on_bad_lines="skip", engine="python"
                )
                # Then select only the columns we want
                available_columns = [
                    col for col in required_columns if col in all_df.columns
                ]
                if available_columns:
                    df = all_df[available_columns]
                    print(
                        f"Successfully loaded CSV by reading all columns first, "
                        f"then selecting: {available_columns}"
                    )
                else:
                    raise ValueError(
                        "None of the required columns found in the CSV file"
                    )
            except Exception as e:
                parsing_errors.append(f"Read all columns then select failed: {e}")

        # If all strategies failed
        if df is None:
            error_msg = (
                f"Failed to parse CSV file {csv_file_path}. Parsing errors:\n"
                + "\n".join(parsing_errors)
            )
            raise pd.errors.ParserError(error_msg)

        # Check which columns we actually have
        available_columns = [col for col in required_columns if col in df.columns]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            print(f"Available columns: {available_columns}")

        if not available_columns:
            raise ValueError(
                f"No required columns found in CSV file. "
                f"Available columns: {list(df.columns)}"
            )

        # Convert timestamp to datetime and sort
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)

        print(f"Successfully loaded {len(df)} records from {csv_file_path}")
        print(f"Selected columns: {list(df.columns)}")

        # Extract configuration parameters if requested
        if any([eval_config_fields, training_param_fields, run_config_fields]):
            print("Extracting configuration parameters...")

            # Pre-populate all requested fields as empty columns
            if eval_config_fields:
                for field in eval_config_fields:
                    df[f"eval_{field}"] = None
                print(
                    f"Added {len(eval_config_fields)} eval_config fields "
                    f"as empty columns"
                )

            if training_param_fields:
                for field in training_param_fields:
                    df[f"training_{field}"] = None
                print(
                    f"Added {len(training_param_fields)} training_param fields "
                    f"as empty columns"
                )

            if run_config_fields:
                for field in run_config_fields:
                    df[f"run_{field}"] = None
                print(
                    f"Added {len(run_config_fields)} run_config fields as empty columns"
                )

            # Now extract and populate the parameters
            df = extract_config_parameters(
                df,
                eval_config_fields=eval_config_fields,
                training_param_fields=training_param_fields,
                run_config_fields=run_config_fields,
            )
            print(
                f"Populated configuration parameters. Total columns: {len(df.columns)}"
            )

        return df

    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file {csv_file_path} is empty")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file {csv_file_path}: {e}")
        print("\nThis might be due to:")
        print("- Fields containing commas that aren't properly quoted")
        print("- Inconsistent number of fields across rows")
        print("- Malformed CSV structure")
        print("\nTry checking the CSV file format or use a different CSV file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV file {csv_file_path}: {e}")
        sys.exit(1)


def prepare_data_for_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for the gradio-leaderboard component.

    Args:
        df: Input DataFrame to prepare

    Returns:
        pd.DataFrame: Prepared DataFrame with formatted columns
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()

    # Remove JSON configuration columns from display
    config_columns_to_hide = ["eval_config", "training_params", "run_config_params"]
    for col in config_columns_to_hide:
        if col in df_copy.columns:
            df_copy = df_copy.drop(columns=[col])

    # Format timestamp for better readability if it exists
    if "timestamp" in df_copy.columns:
        try:
            # Check if timestamp is already datetime
            if pd.api.types.is_datetime64_any_dtype(df_copy["timestamp"]):
                df_copy["timestamp"] = df_copy["timestamp"].dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            else:
                # If it's a string, try to convert and format
                datetime_series = pd.to_datetime(df_copy["timestamp"], errors="coerce")
                # Only format valid timestamps, keep others as strings
                mask = datetime_series.notna()
                df_copy.loc[mask, "timestamp"] = datetime_series[mask].dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                # For invalid timestamps, keep the original string value
        except Exception as e:
            print(f"Warning: Could not format timestamp column: {e}")
            # Keep timestamp as is if formatting fails

    # Rename columns for better display
    column_mapping = {
        "timestamp": "Timestamp",
        "dataset_name": "Dataset",
        "experiment_name": "Experiment",
        "checkpoint_name": "Checkpoint",
        "retrieval_roc_auc": "Retrieval ROC AUC",
        "retrieval_precision_at_1": "Retrieval Precision@1",
        "test_accuracy": "Test Accuracy",
        "test_balanced_accuracy": "Test Balanced Accuracy",
        "test_roc_auc": "Test ROC AUC",
        "test_multiclass_f1": "Test F1 Score",
        "test_map": "Test mAP",
    }

    # Apply column mapping for existing columns
    for old_col, new_col in column_mapping.items():
        if old_col in df_copy.columns:
            df_copy = df_copy.rename(columns={old_col: new_col})

    return df_copy


def parse_config_fields(
    eval_config: str,
    training_params: str,
    run_config_params: str,
    eval_config_fields: Optional[List[str]] = None,
    training_param_fields: Optional[List[str]] = None,
    run_config_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Parse JSON configuration fields and extract specific parameters.

    Args:
        eval_config: JSON string containing evaluation configuration
        training_params: JSON string containing training parameters
        run_config_params: JSON string containing run configuration parameters
        eval_config_fields: List of field names to extract from eval_config
        training_param_fields: List of field names to extract from training_params
        run_config_fields: List of field names to extract from run_config_params

    Returns:
        Dict containing extracted parameters with None values for missing fields
    """
    result = {}

    def extract_from_json(json_str: str, fields: List[str], prefix: str) -> None:
        """Helper function to extract fields from JSON string.

        Args:
            json_str: JSON string to parse
            fields: List of field names to extract
            prefix: Prefix to add to extracted field names
        """
        if not fields or not json_str or json_str.strip() == "":
            return

        try:
            config_data = json.loads(json_str)

            def search_field(
                data: Union[Dict[str, Any], List[Any], str, int, float, bool, None],
                field_name: str,
            ) -> Union[str, int, float, bool, None]:
                """Recursively search for a field in nested dictionary.

                Args:
                    data: Data structure to search in
                    field_name: Name of the field to search for

                Returns:
                    The found value or None if not found
                """
                if isinstance(data, dict):
                    if field_name in data:
                        return data[field_name]
                    for _key, value in data.items():
                        result = search_field(value, field_name)
                        if result is not None:
                            return result
                elif isinstance(data, list):
                    for item in data:
                        result = search_field(item, field_name)
                        if result is not None:
                            return result
                return None

            for field in fields:
                value = search_field(config_data, field)
                result[f"{prefix}_{field}"] = value

        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Failed to parse JSON for {prefix}: {e}")
            # Set all fields to None if JSON parsing fails
            for field in fields:
                result[f"{prefix}_{field}"] = None

    # Extract from eval_config
    if eval_config_fields:
        extract_from_json(eval_config, eval_config_fields, "eval")

    # Extract from training_params
    if training_param_fields:
        extract_from_json(training_params, training_param_fields, "training")

    # Extract from run_config_params
    if run_config_fields:
        extract_from_json(run_config_params, run_config_fields, "run")

    return result


def extract_config_parameters(
    df: pd.DataFrame,
    eval_config_fields: Optional[List[str]] = None,
    training_param_fields: Optional[List[str]] = None,
    run_config_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract configuration parameters from JSON fields in DataFrame.

    Args:
        df: Input DataFrame containing eval_config, training_params,
            run_config_params columns
        eval_config_fields: List of field names to extract from eval_config
        training_param_fields: List of field names to extract from training_params
        run_config_fields: List of field names to extract from run_config_params

    Returns:
        DataFrame with extracted parameters populated in existing columns
    """
    df_copy = df.copy()

    # Check if required columns exist
    required_columns = ["eval_config", "training_params", "run_config_params"]
    missing_columns = [col for col in required_columns if col not in df_copy.columns]

    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        # Add empty columns for missing ones
        for col in missing_columns:
            df_copy[col] = ""

    # Extract parameters for each row
    for idx, row in df_copy.iterrows():
        params = parse_config_fields(
            eval_config=str(row.get("eval_config", "")),
            training_params=str(row.get("training_params", "")),
            run_config_params=str(row.get("run_config_params", "")),
            eval_config_fields=eval_config_fields,
            training_param_fields=training_param_fields,
            run_config_fields=run_config_fields,
        )

        # Populate existing columns with extracted values
        for col_name, value in params.items():
            if col_name in df_copy.columns:
                df_copy.at[idx, col_name] = value

    return df_copy


def main() -> None:
    """Main function to run the Gradio interface."""
    args = parse_arguments()

    # Parse comma-delimited field lists
    eval_config_fields = None
    training_param_fields = None
    run_config_fields = None

    if args.eval_config:
        eval_config_fields = [
            field.strip() for field in args.eval_config.split(",") if field.strip()
        ]

    if args.training_params:
        training_param_fields = [
            field.strip() for field in args.training_params.split(",") if field.strip()
        ]

    if args.run_config_params:
        run_config_fields = [
            field.strip()
            for field in args.run_config_params.split(",")
            if field.strip()
        ]

    # Load data from CSV
    df = load_data(
        args.csv_file,
        eval_config_fields=eval_config_fields,
        training_param_fields=training_param_fields,
        run_config_fields=run_config_fields,
    )

    # Prepare data for leaderboard
    leaderboard_data = prepare_data_for_leaderboard(df)

    # Get unique values for filters
    datasets = ["All"]
    experiments = ["All"]

    if "dataset_name" in df.columns:
        datasets.extend(sorted(df["dataset_name"].unique().tolist()))
    if "experiment_name" in df.columns:
        experiments.extend(sorted(df["experiment_name"].unique().tolist()))

    # Get available metric columns for sorting
    metric_columns = []
    for col in df.columns:
        if col in [
            "retrieval_roc_auc",
            "retrieval_precision_at_1",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_roc_auc",
            "test_multiclass_f1",
            "test_map",
        ]:
            metric_columns.append(col)
        elif df[col].dtype in ["float64", "int64"] and col not in ["timestamp"]:
            metric_columns.append(col)

    # Map metric column names for display
    metric_column_mapping = {
        "retrieval_roc_auc": "Retrieval ROC AUC",
        "retrieval_precision_at_1": "Retrieval Precision@1",
        "test_accuracy": "Test Accuracy",
        "test_balanced_accuracy": "Test Balanced Accuracy",
        "test_roc_auc": "Test ROC AUC",
        "test_multiclass_f1": "Test F1 Score",
        "test_map": "Test mAP",
    }
    display_metric_columns = [
        metric_column_mapping.get(col, col) for col in metric_columns
    ]

    # Create Gradio interface
    with gr.Blocks(title="Experiment Leaderboard") as demo:
        gr.Markdown("# Experiment Leaderboard")
        gr.Markdown(f"**Data Source:** {args.csv_file}")
        gr.Markdown("Track and compare model performance across different experiments")

        with gr.Row():
            dataset_filter = gr.Dropdown(
                choices=datasets, value="All", label="Filter by Dataset"
            )
            experiment_filter = gr.Dropdown(
                choices=experiments, value="All", label="Filter by Experiment"
            )
            metric_sort = gr.Dropdown(
                choices=display_metric_columns,
                value=display_metric_columns[0]
                if display_metric_columns
                else "timestamp",
                label="Sort by Metric",
            )
            refresh_btn = gr.Button("Refresh Data")

        # Create the leaderboard component
        # Define columns to hide from display
        columns_to_hide = ["eval_config", "training_params", "run_config_params"]

        # Get display columns (exclude hidden ones)
        display_columns = [
            col for col in leaderboard_data.columns if col not in columns_to_hide
        ]

        leaderboard = Leaderboard(
            value=leaderboard_data,
            select_columns=display_columns,
            search_columns=["Dataset", "Experiment", "Checkpoint"],
            hide_columns=columns_to_hide,
            filter_columns=["Dataset", "Experiment"],
            interactive=False,
            wrap=True,
            height=600,
        )

        def update_leaderboard(
            dataset_filter: str, experiment_filter: str, metric_sort: str
        ) -> pd.DataFrame:
            """Update the leaderboard based on filters and sorting.

            Args:
                dataset_filter: Selected dataset filter
                experiment_filter: Selected experiment filter
                metric_sort: Selected metric for sorting

            Returns:
                pd.DataFrame: Filtered and sorted DataFrame for display
            """
            filtered_df = df.copy()

            # Apply filters
            if dataset_filter != "All" and "dataset_name" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["dataset_name"] == dataset_filter]
            if experiment_filter != "All" and "experiment_name" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["experiment_name"] == experiment_filter
                ]

            # Sort by selected metric (convert display name back to original
            # column name)
            reverse_mapping = {v: k for k, v in metric_column_mapping.items()}
            original_metric_col = reverse_mapping.get(metric_sort, metric_sort)
            if original_metric_col in filtered_df.columns:
                filtered_df = filtered_df.sort_values(
                    original_metric_col, ascending=False
                )

            # Prepare filtered data for display
            display_df = prepare_data_for_leaderboard(filtered_df)
            return display_df

        # Connect components
        inputs = [dataset_filter, experiment_filter, metric_sort]
        refresh_btn.click(fn=update_leaderboard, inputs=inputs, outputs=leaderboard)
        for component in inputs:
            component.change(fn=update_leaderboard, inputs=inputs, outputs=leaderboard)

        # Initial load
        demo.load(
            fn=lambda *args: update_leaderboard(*args),
            inputs=inputs,
            outputs=leaderboard,
        )

    # Launch the interface
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
