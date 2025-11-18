#!/usr/bin/env python3
"""
Generate Un-aggregated Results Tables
====================================

This script generates well-formatted LaTeX tables of individual dataset results
(not aggregated) with clear visual separations between benchmark groups.

Special handling:
- Individual ID datasets use R-cross-auc when available (more robust)
- BirdNet results are excluded from BirdSet datasets

Creates four tables:
1. Aggregate table - all results across benchmarks
   (excluding clustering for BEANS detection/BirdSet)
2. BEANS table - BEANS Classification + Detection datasets
3. BirdSet table - All BirdSet datasets (BirdNet excluded)
4. Individual ID + Vocal Repertoire table - Combined complex tasks

Usage:
    python generate_unaggregated_tables.py

Output:
    - latex_arxiv/unaggregated_aggregate.tex (All benchmarks)
    - latex_arxiv/unaggregated_beans.tex (BEANS only)
    - latex_arxiv/unaggregated_birdset.tex (BirdSet only)
    - latex_arxiv/unaggregated_complex_tasks.tex (Individual ID + Vocal Repertoire)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts/analysis to path for imports
sys.path.append("scripts/analysis")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input Excel file path
EXCEL_PATH = "~/Downloads/results_with_birdmae.xlsx"

# Define datasets for each benchmark group
BENCHMARK_DATASETS = {
    "BEANS Classification": ["Watkins", "CBI", "HBDB", "BATS", "Dogs", "ESC-50"],
    "BEANS Detection": ["enabirds", "rfcx", "hiceas", "gibbons", "dcase"],
    "BirdSet": ["POW", "PER", "NES", "NBP", "HSN", "SNE", "UHH"],
    "Individual ID": [
        "chiffchaff-cross",
        "littleowls-cross",
        "pipit-cross",
        "macaques",
    ],
    "Vocal Repertoire": [
        "zebrafinch-je-call",
        "Giant_Otters",
        "Bengalese_Finch",
        "SRKW_Orca",
    ],
}

# Define which metrics to include for each dataset type
DATASET_METRICS = {
    # BEANS Classification datasets
    "Watkins": ["Probe", "R-auc", "C-nmi"],
    "CBI": ["Probe", "R-auc", "C-nmi"],
    "HBDB": ["Probe", "R-auc", "C-nmi"],
    "BATS": ["Probe", "R-auc", "C-nmi"],
    "Dogs": ["Probe", "R-auc", "C-nmi"],
    "ESC-50": ["Probe", "R-auc", "C-nmi"],
    # BEANS Detection datasets (exclude clustering)
    "enabirds": ["Probe", "R-auc"],
    "rfcx": ["Probe", "R-auc"],
    "hiceas": ["Probe", "R-auc"],
    "gibbons": ["Probe", "R-auc"],
    "dcase": ["Probe", "R-auc"],
    # BirdSet datasets (exclude clustering)
    "POW": ["Probe", "R-auc"],
    "PER": ["Probe", "R-auc"],
    "NES": ["Probe", "R-auc"],
    "NBP": ["Probe", "R-auc"],
    "HSN": ["Probe", "R-auc"],
    "SNE": ["Probe", "R-auc"],
    "UHH": ["Probe", "R-auc"],
    # Individual ID datasets (special R-auc handling)
    "chiffchaff-cross": ["Probe", "R-auc"],
    "littleowls-cross": ["Probe", "R-auc"],
    "pipit-cross": ["Probe", "R-auc"],
    "macaques": ["Probe", "R-auc"],
    # Vocal Repertoire datasets
    "zebrafinch-je-call": ["R-auc", "C-nmi"],
    "Giant_Otters": ["R-auc", "C-nmi"],
    "Bengalese_Finch": ["R-auc", "C-nmi"],
    "SRKW_Orca": ["R-auc", "C-nmi"],
}

# Models to exclude from tables
EXCLUDE_MODELS = [
    "CLAP",
    "EffNetB0-bio-wabad",
    "EffNetB0-bio-soundscape",
    "EffNetB0-no-whales",
    "EffNetB0-no-birds",
    "EffNetB0-birds",
]

# Visual formatting options
VISUAL_CONFIG = {
    "font_size": "footnotesize",  # tiny, scriptsize, footnotesize, small
    "tabcolsep": "2pt",  # column spacing
    "arraystretch": "1.1",  # row height multiplier
    "group_separator": "|",  # separator between groups
    "decimal_places": 3,  # number of decimal places
}

# Metric caption snippets (un-aggregated tables)
# Use R-AUC and NMI naming, and avoid mentioning clustering where it's not present
UNAGG_METRIC_CAPTIONS = {
    "aggregate": (
        "We report ROC AUC for retrieval as R-AUC; probe accuracy for "
        "BEANS Classification and Individual ID; "
        "mean-average precision for probe on BEANS Detection and BirdSet; "
        "and clustering as NMI where applicable "
        "(BEANS Classification, Vocal Repertoire). "
        "$^{\\dagger}$BirdNet results on BirdSet are excluded following the "
        "authors \\citep{rauchbirdset}."
    ),
    "beans": (
        "We report ROC AUC for retrieval as R-AUC; probe accuracy for "
        "BEANS Classification and "
        "mean-average precision for BEANS Detection. Clustering is reported "
        "as NMI for BEANS Classification only."
    ),
    "birdset": (
        "We report ROC AUC for retrieval as R-AUC and mean-average precision "
        "for probe. "
        "No clustering metrics are reported. "
        "$^{\\dagger}$BirdNet results are excluded following the authors "
        "\\citep{rauchbirdset}."
    ),
    "complex_tasks": (
        "We report ROC AUC for retrieval as R-AUC. Individual ID probe is accuracy; "
        "Vocal Repertoire reports both R-AUC and NMI."
    ),
}

# Note to explain the midrule separating existing vs new models
MIDRULE_NOTE = (
    " Models above the midrule are existing/pretrained checkpoints; "
    "below are new models from this work."
)

# Table configurations
TABLE_CONFIGS = {
    "aggregate": {
        "output_file": "latex_arxiv/unaggregated_aggregate.tex",
        "caption": (
            "Individual dataset results across all bioacoustic benchmarks "
            "(best per metric in bold). "
        )
        + UNAGG_METRIC_CAPTIONS["aggregate"]
        + MIDRULE_NOTE,
        "label": "tab:unaggregated-aggregate",
        "datasets": [],  # Will be populated with all datasets
        "separators": [],  # No separators for aggregate table
        # (models as rows, not datasets)
    },
    "beans": {
        "output_file": "latex_arxiv/unaggregated_beans.tex",
        "caption": (
            "BEANS benchmark results: Classification and Detection tasks "
            "(best per metric in bold). "
        )
        + UNAGG_METRIC_CAPTIONS["beans"]
        + MIDRULE_NOTE,
        "label": "tab:unaggregated-beans",
        "datasets": BENCHMARK_DATASETS["BEANS Classification"]
        + BENCHMARK_DATASETS["BEANS Detection"],
        "separators": ["BEANS Classification"],  # Separator after classification
    },
    # New: Split BEANS tables
    "beans_classification": {
        "output_file": "latex_arxiv/unaggregated_beans_classification.tex",
        "caption": "BEANS Classification datasets only (best per metric in bold). "
        + "We report R-AUC for retrieval; probe accuracy; clustering reported as NMI. "
        + MIDRULE_NOTE,
        "label": "tab:unaggregated-beans-classification",
        "datasets": BENCHMARK_DATASETS["BEANS Classification"],
        "separators": [],
    },
    "beans_detection": {
        "output_file": "latex_arxiv/unaggregated_beans_detection.tex",
        "caption": "BEANS Detection datasets only (best per metric in bold). "
        + "We report R-AUC for retrieval and mean-average precision for probe. "
        + MIDRULE_NOTE,
        "label": "tab:unaggregated-beans-detection",
        "datasets": BENCHMARK_DATASETS["BEANS Detection"],
        "separators": [],
    },
    "birdset": {
        "output_file": "latex_arxiv/unaggregated_birdset.tex",
        "caption": (
            "BirdSet benchmark results: Multi-label bird detection tasks "
            "(best per metric in bold). "
        )
        + UNAGG_METRIC_CAPTIONS["birdset"]
        + MIDRULE_NOTE,
        "label": "tab:unaggregated-birdset",
        "datasets": BENCHMARK_DATASETS["BirdSet"],
        "separators": [],
    },
    "complex_tasks": {
        "output_file": "latex_arxiv/unaggregated_complex_tasks.tex",
        "caption": (
            "Complex bioacoustic tasks: Individual ID and Vocal Repertoire "
            "analysis (best per metric in bold). "
        )
        + UNAGG_METRIC_CAPTIONS["complex_tasks"]
        + MIDRULE_NOTE,
        "label": "tab:unaggregated-complex",
        "datasets": BENCHMARK_DATASETS["Individual ID"]
        + BENCHMARK_DATASETS["Vocal Repertoire"],
        "separators": ["Individual ID"],  # Separator after Individual ID
    },
}


# =============================================================================
# FUNCTIONS
# =============================================================================


def load_excel_data() -> pd.DataFrame:
    """Load and process Excel data.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with Excel data.
    """
    from visualize_excel import load_excel

    raw = load_excel(EXCEL_PATH)
    print(f"📋 Raw data shape: {raw.shape}")

    # Filter out excluded models
    # Check if we have a model name column (usually 'Unnamed: 1' contains model names)
    model_col = "Unnamed: 1"
    if model_col in raw.columns:
        # Create mask for models to exclude
        mask = ~raw[model_col].astype(str).str.contains(
            "|".join(EXCLUDE_MODELS), na=False
        )
        filtered = raw[mask]
    else:
        # If index contains model names, convert to string
        mask = ~raw.index.astype(str).str.contains("|".join(EXCLUDE_MODELS), na=False)
        filtered = raw[mask]

    print(f"📋 Filtered data shape: {filtered.shape}")
    return filtered


def get_dataset_columns(
    df: pd.DataFrame, datasets: list[str]
) -> tuple[list[str], list[tuple[str, int, list[str]]]]:
    """Get available columns for specified datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    datasets : list[str]
        List of dataset names.

    Returns
    -------
    tuple[list[str], list[tuple[str, int, list[str]]]]
        A tuple containing available columns and dataset info.
    """
    available_columns = []
    dataset_info = []

    for dataset in datasets:
        if dataset not in DATASET_METRICS:
            print(f"Warning: No metrics defined for dataset {dataset}")
            continue

        metrics = DATASET_METRICS[dataset]
        dataset_cols = []

        for metric in metrics:
            # Special handling for Individual ID R-auc (prefer R-cross-auc)
            if dataset in BENCHMARK_DATASETS["Individual ID"] and metric == "R-auc":
                cross_key = f"{dataset}_R-cross-auc"
                regular_key = f"{dataset}_R-auc"

                if cross_key in df.columns:
                    available_columns.append(cross_key)
                    dataset_cols.append("R-auc*")  # Mark as cross-validation
                    print(f"  Using {cross_key} for {dataset}")
                elif regular_key in df.columns:
                    available_columns.append(regular_key)
                    dataset_cols.append("R-auc")
                    print(f"  Using {regular_key} for {dataset}")
            else:
                # Standard metric handling
                key = f"{dataset}_{metric}"
                if key in df.columns:
                    available_columns.append(key)
                    dataset_cols.append(metric)

        if dataset_cols:
            dataset_info.append((dataset, len(dataset_cols), dataset_cols))

    return available_columns, dataset_info


def apply_bold_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """Apply bold formatting to maximum values in each column.

    Includes special N/A handling for BirdNet on BirdSet.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with bold formatting applied.
    """
    fmt = df.copy().astype(str)

    # Get BirdSet dataset names for checking
    birdset_datasets = BENCHMARK_DATASETS["BirdSet"]

    for col in df.columns:
        maxv = df[col].max(skipna=True)
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                # Special case: BirdNet on BirdSet datasets gets "N/A"
                is_birdnet = "BirdNet" in idx
                is_birdset_col = any(ds in col for ds in birdset_datasets)

                if is_birdnet and is_birdset_col:
                    fmt.at[idx, col] = "N/A"
                else:
                    fmt.at[idx, col] = "---"
            else:
                if abs(val - maxv) < 1e-6:
                    fmt.at[idx, col] = (
                        f"\\textbf{{{val:.{VISUAL_CONFIG['decimal_places']}f}}}"
                    )
                else:
                    fmt.at[idx, col] = f"{val:.{VISUAL_CONFIG['decimal_places']}f}"

    return fmt


def get_benchmark_group(dataset: str) -> str:
    """Get the benchmark group name for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name.

    Returns
    -------
    str
        Benchmark group name, or "Unknown" if not found.
    """
    for group_name, datasets in BENCHMARK_DATASETS.items():
        if dataset in datasets:
            return group_name
    return "Unknown"


def build_latex_table(
    fmt_df: pd.DataFrame,
    dataset_info: list[tuple[str, int, list[str]]],
    separators_after: list[str],
    caption: str,
    label: str,
) -> str:
    """Build the complete LaTeX table with dataset groupings.

    Parameters
    ----------
    fmt_df : pd.DataFrame
        Formatted dataframe.
    dataset_info : list[tuple[str, int, list[str]]]
        List of (dataset_name, col_count, metrics) tuples.
    separators_after : list[str]
        List of dataset names to add separators after.
    caption : str
        Table caption.
    label : str
        Table label.

    Returns
    -------
    str
        LaTeX table content.
    """

    # Clean column names and index
    fmt_df.columns = [c.replace("_", "-").replace("@", "\\@") for c in fmt_df.columns]
    fmt_df.index = [i.replace("_", "-") for i in fmt_df.index]

    # Build column format
    colfmt = "l|"  # model column with separator

    for i, (_, col_count, _) in enumerate(dataset_info):
        colfmt += "c" * col_count
        if i < len(dataset_info) - 1:  # Add separator except after last dataset
            colfmt += "|"

    # Header row 1: dataset names
    header1 = [""]  # empty for model column
    for dataset_name, col_count, _ in dataset_info:
        formatted_name = f"\\textbf{{{dataset_name}}}"
        if col_count > 1:
            header1.append(f"\\multicolumn{{{col_count}}}{{c}}{{{formatted_name}}}")
        else:
            header1.append(formatted_name)

    # Header row 2: metric names (display mapping: R-auc -> R-AUC, C-nmi -> NMI)
    def _display_metric(m: str) -> str:
        if m == "R-auc":
            return "R-AUC"
        if m == "C-nmi":
            return "NMI"
        if m == "R-auc*" or m == "ROC AUC*":
            return "R-AUC*"
        return m

    header2 = ["\\textbf{Model}"]
    for _dataset_name, _, metrics in dataset_info:
        for m in metrics:
            pretty = _display_metric(m)
            clean_m = pretty.replace("@", "\\@")
            header2.append(f"\\textit{{{clean_m}}}")

    # Build partial horizontal lines under dataset headers
    cline_commands = []
    col_start = 2  # Start after model column
    for _dataset_name, col_count, _ in dataset_info:
        if col_count > 1:  # Only add cline for multi-column headers
            col_end = col_start + col_count - 1
            cline_commands.append(f"\\cline{{{col_start}-{col_end}}}")
        col_start += col_count

    # Build table body with benchmark group separators
    lines = ["\\toprule"]
    lines.append(" & ".join(header1) + " \\\\")
    if cline_commands:
        lines.extend(cline_commands)
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")

    for _i, (idx, row) in enumerate(fmt_df.iterrows()):
        line_parts = [f"\\textbf{{{idx}}}"] + list(row.values)
        lines.append(" & ".join(line_parts) + " \\\\")

        # Add separator if we're at the end of a benchmark group
        if separators_after:
            # Find which dataset this row represents (this is for dataset-based tables)
            # For model-based tables, we add separators at specific model names
            if idx in separators_after:
                lines.append("\\midrule")

    lines.append("\\bottomrule")
    body = "\n".join(lines)

    # Complete LaTeX table (caption above table)
    latex_content = f"""
\\begin{{table*}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\{VISUAL_CONFIG["font_size"]}
\\setlength{{\\tabcolsep}}{{{VISUAL_CONFIG["tabcolsep"]}}}
\\renewcommand{{\\arraystretch}}{{{VISUAL_CONFIG["arraystretch"]}}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{colfmt}}}
{body}
\\end{{tabular}}%
}}
\\end{{table*}}
"""

    return latex_content


def generate_table(df: pd.DataFrame, table_config: dict) -> str:
    """Generate a single table based on configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    table_config : dict
        Table configuration dictionary.

    Returns
    -------
    str
        LaTeX table content.
    """
    datasets = table_config["datasets"]
    output_file = table_config["output_file"]
    caption = table_config["caption"]
    label = table_config["label"]
    separators = table_config.get("separators", [])

    # Get columns for specified datasets
    cols, dataset_info = get_dataset_columns(df, datasets)

    if not cols:
        print(f"Warning: No columns found for table {output_file}")
        return None

    # Select data
    selected_df = df[cols].round(VISUAL_CONFIG["decimal_places"])

    # Special handling: exclude BirdNet results on BirdSet datasets
    birdset_datasets = BENCHMARK_DATASETS["BirdSet"]
    birdset_cols = [
        col for col in selected_df.columns if any(ds in col for ds in birdset_datasets)
    ]

    if birdset_cols:
        # Handle potential case variations or exact name matching
        birdnet_rows = selected_df.index[
            selected_df.index.str.contains("BirdNet", case=False, na=False)
        ]
        if len(birdnet_rows) > 0:
            for col in birdset_cols:
                selected_df.loc[birdnet_rows, col] = np.nan
            print(
                f"  Excluding BirdNet results for BirdSet datasets: "
                f"{list(birdnet_rows)} on {len(birdset_cols)} columns"
            )
        else:
            print("  BirdNet not found in index for BirdSet exclusion")

    # Apply formatting
    formatted_df = apply_bold_formatting(selected_df)

    # Build separators after specified groups/datasets
    separators_after = []
    if separators:
        # For the aggregate table, we need to find where each group ends
        if "aggregate" in output_file:
            # Group datasets by benchmark and find last dataset in each separator group
            for sep_group in separators:
                if sep_group in BENCHMARK_DATASETS:
                    group_datasets = BENCHMARK_DATASETS[sep_group]
                    # Find the last dataset from this group that's actually in our table
                    for dataset in reversed(group_datasets):
                        if dataset in datasets:
                            separators_after.append(dataset)
                            break
        else:
            separators_after = separators

    # Always add a horizontal midrule split between existing vs new models.
    # We place it after the last "existing" model present, ideally after Perch.
    existing_priority = [
        "Perch",
        "SurfPerch",
        "BirdNet",
        "BEATS (SFT)",
        "BEATS (pretrained)",
        "EAT-base (pretrained)",
        "EAT-base (SFT)",
        "Bird-AVES-biox-base",
        "BEATS-NatureLM-audio",
    ]
    present = selected_df.index.tolist()
    split_after = None
    for name in existing_priority:
        if name in present:
            split_after = name
            break
    if split_after:
        separators_after = separators_after + [split_after]

    # Generate LaTeX table
    latex_content = build_latex_table(
        formatted_df, dataset_info, separators_after, caption, label
    )

    # Save to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(latex_content)

    print(f"✅ Generated table: {output_file}")
    print(f"   Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"   Columns: {len(cols)}")
    print(f"   Models: {len(formatted_df)}")

    return latex_content


def main() -> None:
    """Main function to generate all un-aggregated tables."""
    print("🔄 Loading Excel data...")
    df = load_excel_data()

    # Populate aggregate table with all datasets
    all_datasets = []
    for group_datasets in BENCHMARK_DATASETS.values():
        all_datasets.extend(group_datasets)
    TABLE_CONFIGS["aggregate"]["datasets"] = all_datasets

    # Generate all tables
    print("\n📊 Generating un-aggregated tables...")

    for table_name, table_config in TABLE_CONFIGS.items():
        print(f"\n🔄 Processing {table_name} table...")
        try:
            generate_table(df, table_config)
        except Exception as e:
            print(f"❌ Error generating {table_name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n🎉 All un-aggregated tables generated successfully!")
    for table_name, config in TABLE_CONFIGS.items():
        print(f"📁 {table_name}: {config['output_file']}")


if __name__ == "__main__":
    main()
