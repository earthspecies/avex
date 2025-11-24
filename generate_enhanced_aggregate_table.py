#!/usr/bin/env python3
"""
Generate Enhanced Aggregate Results Table
=========================================

This script generates a well-formatted LaTeX table of aggregate benchmark results
with clear visual separations between benchmark groups.

Special handling for Individual ID: Reports R-auc and Probe columns, where R-auc
uses R-cross-auc values when available (as they are more robust to confounds).

Usage:
    python generate_enhanced_aggregate_table.py

Output:
    - aggregate_enhanced_visual.tex (LaTeX table file)
"""

import ast
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

# Output LaTeX file
OUTPUT_FILE = "latex_arxiv/aggregate_v0.tex"
OUTPUT_FILE_NOESC50 = "latex_arxiv/aggregate_v0_noesc50.tex"
OUTPUT_FILE_HIGHLIGHTED = "latex_arxiv/aggregate_v0_highlighted.tex"

# Define which metrics to include for each benchmark group
# Keep internal metric keys as in the data (R-auc, C-nmi),
# and map to AUROC/nmi only for display
GROUP_METRICS = {
    "BEANS Classification": ["Probe", "R-auc", "C-nmi"],
    "BEANS Detection": ["Probe", "R-auc"],
    "BirdSet": ["Probe", "R-auc"],
    "Individual ID": ["Probe", "R-auc"],
    "Vocal Repertoire": ["R-auc", "C-nmi"],
}

# Override group datasets to ensure correct groupings
CORRECTED_GROUP_DATASETS = {
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

# Models to exclude from the table
EXCLUDE_MODELS = [
    "CLAP",
    "EffNetB0-bio-wabad",
    "EffNetB0-bio-soundscape",
    "EffNetB0-no-whales",
    "EffNetB0-no-birds",
    "EffNetB0-birds",
]

# Shared caption suffix used in captions to maintain consistency
METRIC_CAPTION_SUFFIX = (
    "We report ROC AUC for retrieval, accuracy for probing on BEANS "
    "classification and Individual ID, "
    r"mean-average precision for probe on BEANS Detection and BirdSet. "
    r"We report the mean of each metric over datasets per benchmark. "
    r"$^{\dagger}$BirdNet results on BirdSet are excluded following the "
    r"authors \citep{rauchbirdset} due to data leakage"
)

# Define sub-tables with model groupings and separators
SUB_TABLES = {
    "efficientnet_analysis": {
        "models": [
            # Pretrained models first
            "Perch",
            "SurfPerch",
            "BirdNet",
            # Separator after pretrained models
            "---SEPARATOR---",
            # Our EffNet models (AudioSet, bio, all)
            "EffNetB0-AudioSet",
            "EffNetB0-bio",
            "EffNetB0-all",
        ],
        "output_file": "latex/efficientnet_analysis.tex",
        "alt_output_file": "latex_arxiv/efficientnet_analysis.tex",
        "caption": (
            "Comparison of EfficientNet models trained on mixes of AudioSet "
            "and large-scale bioacoustic data. " + METRIC_CAPTION_SUFFIX
        ),
        "show_training_tags": False,
        "append_dagger_to": ["BirdNet"],
        "label": "tab:efficientnet-analysis",
    },
    "ssl_models_analysis": {
        "models": [
            # Our EAT models only (remove sl- post-trained models)
            # Pretrained SSL models
            "BEATS (SFT)",
            "BEATS (pretrained)",
            "Bird-AVES-biox-base",
            # Exclude NatureLM-audio as requested
            "---SEPARATOR---",
            "EAT-AS",
            "EAT-bio",
            "EAT-all",
            # Separator after our models
        ],
        "output_file": "latex/ssl_models_analysis.tex",
        "caption": ("SSL model analysis: our trained models vs. pretrained SSL baselines."),
        "label": "tab:ssl-analysis",
    },
}

# Visual formatting options
VISUAL_CONFIG = {
    "font_size": "footnotesize",  # tiny, scriptsize, footnotesize, small
    "tabcolsep": "3pt",  # column spacing
    "arraystretch": "1.2",  # row height multiplier
    "group_separator": "||",  # separator between groups (| or ||)
    "decimal_places": 3,  # number of decimal places
}

TABLE_CAPTION = (
    "Aggregate results across bioacoustic benchmarks and tasks "
    "(best per metric in bold). "
    + METRIC_CAPTION_SUFFIX
    + "Model labels carry training tags: \\textsuperscript{SSL} "
    "self-supervised, \\textsuperscript{SL} supervised, "
    "\\textsuperscript{SL-SSL} supervised fine-tuning after SSL pretraining. "
    "Models above the midrule are existing/pretrained checkpoints; "
    "below are new models from this work."
)
TABLE_LABEL = "tab:aggregate_results"

# =============================================================================
# SCRIPT
# =============================================================================


def extract_groups_and_metrics() -> tuple[dict, list]:
    """Extract metrics from visualize_excel.py, but use our corrected group datasets.

    Returns:
        tuple[dict, list]: A tuple containing the corrected group datasets
            and the metrics list.
    """
    code = Path("scripts/analysis/visualize_excel.py").read_text()
    main = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "main"][
        0
    ]

    metrics = None

    for node in main.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    if t.id == "metrics":
                        metrics = ast.literal_eval(node.value)

    # Use our corrected group datasets instead of the original
    return CORRECTED_GROUP_DATASETS, metrics


def get_group_datasets_noesc50() -> dict:
    """Get group datasets with ESC-50 removed.

    Returns:
        dict: Group datasets dictionary with ESC-50 removed from all groups.
    """
    noesc50_datasets = {}
    for group_name, datasets in CORRECTED_GROUP_DATASETS.items():
        noesc50_datasets[group_name] = [ds for ds in datasets if ds != "ESC-50"]
    return noesc50_datasets


def validate_dataset_coverage(df: pd.DataFrame, group_datasets: dict, metrics: list) -> None:
    """Validate that all expected datasets exist in the data and report missing ones.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to validate.
    group_datasets : dict
        Dictionary mapping group names to lists of dataset names.
    metrics : list
        List of metric names to check.

    Returns
    -------
    None
    """
    print("\n🔍 Validating dataset coverage...")

    all_missing = []
    all_found = []

    for group_name, datasets in group_datasets.items():
        print(f"\n📋 {group_name}:")
        group_missing = []
        group_found = []

        for dataset in datasets:
            # Check if any metric columns exist for this dataset
            dataset_cols = [col for col in df.columns if col.startswith(f"{dataset}_")]

            if dataset_cols:
                group_found.append(dataset)
                print(f"  ✅ {dataset}: {len(dataset_cols)} columns found")

                # Check which specific metrics are available
                for metric in metrics:
                    expected_col = f"{dataset}_{metric}"
                    if expected_col in df.columns:
                        print(f"     📊 {metric}: ✓")
                    else:
                        print(f"     📊 {metric}: ❌ missing")
            else:
                group_missing.append(dataset)
                print(f"  ❌ {dataset}: No columns found")

        all_missing.extend([(group_name, ds) for ds in group_missing])
        all_found.extend([(group_name, ds) for ds in group_found])

    print("\n📊 Summary:")
    print(f"  Found datasets: {len(all_found)}")
    print(f"  Missing datasets: {len(all_missing)}")

    if all_missing:
        print("\n❌ Missing datasets:")
        for group, dataset in all_missing:
            print(f"  {group}: {dataset}")

    return all_missing, all_found


def compute_group_averages_with_prioritization(
    df: pd.DataFrame,
    group_datasets: dict,
    metrics: list,
) -> pd.DataFrame:
    """Compute per-model averages with special handling.

    Handles Individual ID cross-retrieval and BirdNet exclusion on BirdSet.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with model results.
    group_datasets : dict
        Dictionary mapping group names to lists of dataset names.
    metrics : list
        List of metrics to compute averages for.

    Returns
    -------
    pd.DataFrame
        Dataframe with computed group averages.
    """
    result = pd.DataFrame(index=df.index)

    for group_name, datasets in group_datasets.items():
        print(f"\nProcessing {group_name} with datasets: {datasets}")

        # Get metrics for this specific group
        group_metrics = metrics.get(group_name, []) if isinstance(metrics, dict) else metrics

        for metric in group_metrics:
            if group_name == "Individual ID" and metric == "R-auc":
                # Special handling: prefer R-cross-auc over R-auc when available
                prioritized_values = []

                for dataset in datasets:
                    cross_key = f"{dataset}_R-cross-auc"
                    regular_key = f"{dataset}_R-auc"

                    if cross_key in df.columns:
                        # Use cross-auc when available (more robust)
                        prioritized_values.append(df[cross_key])
                        print(f"  Using {cross_key} for {dataset}")
                    elif regular_key in df.columns:
                        # Fall back to regular R-auc
                        prioritized_values.append(df[regular_key])
                        print(f"  Using {regular_key} for {dataset}")

                if prioritized_values:
                    # Combine all prioritized values and compute mean
                    combined_df = pd.concat(prioritized_values, axis=1)

                    # Check for NaN values before averaging
                    nan_counts = combined_df.isna().sum()
                    total_cells = len(combined_df) * len(combined_df.columns)
                    total_nans = nan_counts.sum()

                    if total_nans > 0:
                        print(
                            f"  ⚠️  Found {total_nans}/{total_cells} empty cells, "
                            "excluding from average"
                        )
                        for col, nan_count in nan_counts.items():
                            if nan_count > 0:
                                print(f"     {col}: {nan_count} empty")

                    avg = combined_df.mean(axis=1, skipna=True)
                    result[f"{group_name}_{metric}"] = avg
                    print(f"  Individual ID R-auc: Combined {len(prioritized_values)} datasets")

            else:
                # Standard aggregation for all other cases
                keys = [f"{ds}_{metric}" for ds in datasets if f"{ds}_{metric}" in df.columns]
                missing_keys = [
                    f"{ds}_{metric}" for ds in datasets if f"{ds}_{metric}" not in df.columns
                ]

                if missing_keys:
                    print(f"  ❌ Missing columns for {group_name} {metric}: {missing_keys}")

                if keys:
                    values = df[keys].apply(pd.to_numeric, errors="coerce")

                    # Special handling: exclude BirdNet results on BirdSet datasets
                    if group_name == "BirdSet":
                        # Handle potential case variations or exact name matching
                        birdnet_rows = values.index[
                            values.index.str.contains("BirdNet", case=False, na=False)
                        ]
                        if len(birdnet_rows) > 0:
                            # Store the exclusion info before computing average
                            birdnet_exclusion = list(birdnet_rows)
                            values.loc[birdnet_rows] = np.nan
                            print(
                                f"  🚫 Excluding BirdNet results for BirdSet datasets: "
                                f"{birdnet_exclusion}"
                            )
                        else:
                            print("  ℹ️  BirdNet not found in index for BirdSet exclusion")

                    # Check for NaN values before averaging
                    nan_counts = values.isna().sum()
                    total_cells = len(values) * len(values.columns)
                    total_nans = nan_counts.sum()

                    if total_nans > 0:
                        print(
                            f"  ⚠️  Found {total_nans}/{total_cells} empty cells, "
                            "excluding from average"
                        )
                        # Report which models have missing data
                        models_with_nans = values.isna().any(axis=1)
                        if models_with_nans.any():
                            nan_models = values.index[models_with_nans].tolist()
                            print(f"     Models with missing data: {nan_models}")

                    avg = values.mean(axis=1, skipna=True)
                    result[f"{group_name}_{metric}"] = avg
                    print(
                        f"  ✅ {group_name} {metric}: Found {len(keys)}/"
                        f"{len(datasets)} datasets - {keys}"
                    )
                else:
                    print(f"  ❌ No data found for {group_name} {metric}")

    return result


def load_and_process_data(group_datasets: dict | None = None) -> pd.DataFrame:
    """Load Excel data and compute group averages with prioritization.

    Parameters
    ----------
    group_datasets : dict | None, optional
        Dictionary mapping group names to lists of dataset names.
        If None, will be extracted from visualize_excel.py.

    Returns
    -------
    pd.DataFrame
        Processed dataframe with group averages computed.
    """
    from visualize_excel import load_excel

    if group_datasets is None:
        group_datasets, metrics = extract_groups_and_metrics()
    else:
        _, metrics = extract_groups_and_metrics()

    # Load raw data
    raw = load_excel(EXCEL_PATH)

    print(f"📋 Raw data shape: {raw.shape}")
    print(f"🏷️  Available columns: {len(raw.columns)} total")

    # Show corrected groupings
    print(f"🔧 BEANS Classification: {group_datasets['BEANS Classification']}")
    print(f"🔧 BirdSet: {group_datasets['BirdSet']}")

    # Validate dataset coverage before processing
    missing_datasets, found_datasets = validate_dataset_coverage(raw, group_datasets, metrics)

    # Compute group averages with special Individual ID handling
    print("\n🔄 Computing group averages with cross-retrieval prioritization...")
    full = compute_group_averages_with_prioritization(raw, group_datasets, metrics)

    # Remove clustering ARI columns, keep NMI
    full = full.drop(columns=[c for c in full.columns if c.endswith("C-ari")], errors="ignore")
    full = full.round(VISUAL_CONFIG["decimal_places"])

    # Filter out specified models
    # Convert index to string to handle non-string indices
    mask = ~full.index.astype(str).str.contains("|".join(EXCLUDE_MODELS), na=False)
    full = full[mask]

    return full


def build_column_structure(
    df: pd.DataFrame,
    include_groups: list[str] | None = None,
    comparison_df: pd.DataFrame | None = None,
) -> tuple[list[str], list[tuple[str, int, list[str]]]]:
    """Build column structure based on GROUP_METRICS configuration.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with aggregated results.
    include_groups : list[str] | None, optional
        Optional list of group names to include. If None, includes all groups.
    comparison_df : pd.DataFrame | None, optional
        Optional DataFrame for side-by-side comparison (used for _noesc50 tables).

    Returns
    -------
    tuple[list[str], list[tuple[str, int, list[str]]]]
        A tuple containing the list of column names and group information.
    """
    cols = []
    group_info = []

    groups_to_process = include_groups if include_groups is not None else GROUP_METRICS.keys()

    for group_name in groups_to_process:
        if group_name not in GROUP_METRICS:
            continue
        metrics = GROUP_METRICS[group_name]
        g_cols = []

        # For BEANS Classification with comparison_df, show both versions side-by-side
        if group_name == "BEANS Classification" and comparison_df is not None:
            # First add columns from df (without ESC-50) with suffix
            for m in metrics:
                key = f"{group_name}_{m}"
                if key in df.columns:
                    cols.append(f"{key}_noesc50")
                    g_cols.append(m)
                else:
                    print(f"Warning: {key} not found in columns (no ESC-50)")

            # Then add columns from comparison_df (with ESC-50) with suffix
            for m in metrics:
                key = f"{group_name}_{m}"
                if key in comparison_df.columns:
                    cols.append(f"{key}_wesc50")
                else:
                    print(f"Warning: {key} not found in columns (with ESC-50)")

            if g_cols:
                # Create two group entries: one for each version
                group_info.append((f"{group_name} (w/o ESC-50)", len(g_cols), g_cols))
                group_info.append((f"{group_name} (w/ ESC-50)", len(g_cols), g_cols))
        else:
            # Normal case: single version
            for m in metrics:
                key = f"{group_name}_{m}"
                if key in df.columns:
                    cols.append(key)
                    g_cols.append(m)
                else:
                    print(f"Warning: {key} not found in columns")

            if g_cols:
                group_info.append((group_name, len(g_cols), g_cols))

    return cols, group_info


def _determine_model_type(raw_name: str) -> str:
    """Return one of {'SSL', 'SL', 'SL-SSL'} based on model naming patterns.

    We standardize ambiguous 'SSL/SL' wording to 'SL-SSL'.

    Returns
    -------
    str
        One of 'SSL', 'SL', or 'SL-SSL'.
    """
    name = raw_name or ""
    lower = name.lower()

    # Explicit rules first
    if lower in {"beats-naturelm-audio", "naturebeats"}:
        return "SL-SSL"

    # sl- models: supervised fine-tuning of SSL backbones
    if lower.startswith("sl-beats") or lower.startswith("sl-eat"):
        return "SL-SSL"

    # Our EAT SSL pretrains
    if lower in {"eat-as", "eat-bio", "eat-all"}:
        return "SSL"

    # EAT-base variants
    if lower.startswith("eat-base"):
        if "(sft)" in lower:
            return "SL-SSL"
        return "SSL"

    # BEATS existing checkpoints
    if lower.startswith("beats"):
        return "SSL"

    # Bird-AVES SSL baseline
    if lower.startswith("bird-aves"):
        return "SSL"

    # EffNets are supervised
    if lower.startswith("effnet"):
        return "SL"

    # Classic supervised baselines
    if lower in {"perch", "surfperch", "birdnet"}:
        return "SL"

    # Default fallback (be explicit rather than silent)
    return "SSL"


def _rename_model_for_display(raw_name: str) -> str:
    """Apply explicit display-name overrides (e.g., NatureBEATs).

    Parameters
    ----------
    raw_name : str
        The raw model name.

    Returns
    -------
    str
        The display name (overridden if in rename_map, otherwise raw_name).
    """
    rename_map = {
        "BEATS-NatureLM-audio": "NatureBEATs",
    }
    # Check for BirdMAE variants (case-insensitive)
    if "birdmae" in raw_name.lower() or "bird-mae" in raw_name.lower():
        return "Bird-MAE-Huge"
    return rename_map.get(raw_name, raw_name)


def decorate_model_names_for_table(
    df: pd.DataFrame,
    *,
    show_training_tags: bool = True,
    append_dagger_to: list | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return a copy of df where the index is replaced with display names.

    Includes training tags in the display names.

    Also return a mapping from raw name -> decorated display name for downstream use
    (e.g., to place separators after a specific model).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    show_training_tags : bool, optional
        Whether to show training tags in display names. Defaults to True.
    append_dagger_to : list | None, optional
        List of model names to append dagger symbol to. Defaults to None.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A tuple containing the dataframe with decorated names and a mapping
        from raw names to decorated names.
    """
    display_names = []
    mapping = {}
    append_dagger_to = append_dagger_to or []
    for raw in df.index.tolist():
        renamed = _rename_model_for_display(raw)
        decorated = renamed
        if any(raw == target or renamed == target for target in append_dagger_to):
            decorated = f"{decorated}$^\\dagger$"
        if show_training_tags:
            tag = _determine_model_type(raw)
            decorated = f"{decorated}\\textsuperscript{{{tag}}}"
        display_names.append(decorated)
        mapping[raw] = decorated

    out = df.copy()
    out.index = display_names
    return out, mapping


def reorder_models_with_separator(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Reorder models so that existing/pretrained models come first, then new models.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The reordered frame and the raw-name of the last 'existing' model
        to place a midrule after.
    """
    raw_models = df.index.tolist()

    preferred_existing = [
        "BEATS (SFT)",
        "BEATS (pretrained)",
        "EAT-base (pretrained)",
        "EAT-base (SFT)",
        "Bird-AVES-biox-base",
        "BEATS-NatureLM-audio",  # will be shown as NatureBEATs
        "BirdMAE (pretrained)",  # will be shown as Bird-MAE-Huge
        "SurfPerch",
        "BirdNet",
        "Perch",
    ]

    preferred_new = [
        "EffNetB0-AudioSet",
        "EffNetB0-bio",
        "EffNetB0-all",
        "EAT-AS",
        "EAT-bio",
        "EAT-all",
        "sl-BEATS-bio",
        "sl-BEATS-all",
        "sl-EAT-bio",
        "sl-EAT-all",
    ]

    def is_existing(name: str) -> bool:
        # Ensure name is a string
        if not isinstance(name, str):
            return False
        name_lower = name.lower()
        if name_lower in {"perch", "surfperch", "birdnet"}:
            return True
        if name_lower.startswith("beats") or name_lower.startswith("bird-aves"):
            return True
        if name_lower.startswith("eat-base"):
            return True
        if name_lower == "beats-naturelm-audio":
            return True
        if "birdmae" in name_lower or "bird-mae" in name_lower:
            return True
        return False

    ordered = []

    # Filter out non-string model names
    raw_models = [m for m in raw_models if isinstance(m, str)]

    # Add preferred existing in order
    for m in preferred_existing:
        if m in raw_models:
            ordered.append(m)

    # Add any other existing not yet included
    for m in raw_models:
        if m not in ordered and is_existing(m):
            ordered.append(m)

    # Remember the last existing model for the separator;
    # if none, place after first group by default
    last_existing = ordered[-1] if ordered else None

    # Add preferred new models in order
    for m in preferred_new:
        if m in raw_models and m not in ordered:
            ordered.append(m)

    # Add any remaining models (treat as new)
    for m in raw_models:
        if m not in ordered:
            ordered.append(m)

    # Reindex to this order intersecting with current
    df_reordered = df.reindex([m for m in ordered if m in df.index])
    return df_reordered, last_existing or "Perch"


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
        Dataframe with bold formatting applied to maximum values.
    """
    fmt = df.copy().astype(str)

    for col in df.columns:
        maxv = df[col].max(skipna=True)
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                # Special case: BirdNet on BirdSet datasets gets "N/A"
                if "BirdNet" in idx and "BirdSet" in col:
                    fmt.at[idx, col] = "N/A"
                else:
                    fmt.at[idx, col] = "---"
            else:
                if abs(val - maxv) < 1e-6:
                    fmt.at[idx, col] = f"\\textbf{{{val:.{VISUAL_CONFIG['decimal_places']}f}}}"
                else:
                    fmt.at[idx, col] = f"{val:.{VISUAL_CONFIG['decimal_places']}f}"

    return fmt


def build_latex_table(
    fmt_df: pd.DataFrame,
    group_info: list[tuple[str, int, list[str]]],
    separators_after: list[str] | None = None,
    caption: str | None = None,
    label: str | None = None,
    resize_width: str | None = None,
) -> tuple[str, str]:
    """Build the complete LaTeX table with optional separators.

    Parameters
    ----------
    fmt_df : pd.DataFrame
        Formatted DataFrame.
    group_info : list[tuple[str, int, list[str]]]
        List of (group_name, col_count, metrics) tuples.
    separators_after : list[str] | None, optional
        List of model names to add separators after. Defaults to None.
    caption : str | None, optional
        Table caption. Defaults to None.
    label : str | None, optional
        Table label. Defaults to None.
    resize_width : str | None, optional
        Width for resizebox (e.g., '\\textwidth', '0.6\\textwidth',
        or None to disable resizebox). Defaults to None.

    Returns
    -------
    tuple[str, str]
        A tuple containing the LaTeX content and column format string.
    """

    # Use defaults if not provided
    if caption is None:
        caption = TABLE_CAPTION
    if label is None:
        label = TABLE_LABEL
    if separators_after is None:
        separators_after = []
    if resize_width is None:
        resize_width = "\\textwidth"

    # Clean column names and index (also display mapping AUROC/nmi)
    def _display_col(name: str) -> str:
        name = name.replace("_", "-")
        name = name.replace("@", "\\@")
        name = name.replace("R-auc", "R-AUC")
        name = name.replace("C-nmi", "nmi")
        return name

    fmt_df.columns = [_display_col(c) for c in fmt_df.columns]
    fmt_df.index = [i.replace("_", "-") for i in fmt_df.index]

    # Build column format with separators
    sep = VISUAL_CONFIG["group_separator"]
    colfmt = f"l{sep}"  # model column with separator

    for i, (_, col_count, _) in enumerate(group_info):
        colfmt += "c" * col_count
        if i < len(group_info) - 1:  # Add separator except after last group
            colfmt += sep

    # Header row 1: multicolumn for groups
    header1 = [""]  # empty for model column
    for group_name, col_count, _ in group_info:
        formatted_name = f"\\textbf{{{group_name}}}"
        header1.append(f"\\multicolumn{{{col_count}}}{{c}}{{{formatted_name}}}")

    # Header row 2: metric names
    header2 = ["\\textbf{Model}"]
    for _group_name, _, metrics in group_info:
        for m in metrics:
            clean_m = m.replace("@", "\\@")
            header2.append(f"\\textit{{{clean_m}}}")

    # Build partial horizontal lines under group headers
    cline_commands = []
    col_start = 2  # Start after model column
    for _group_name, col_count, _ in group_info:
        col_end = col_start + col_count - 1
        cline_commands.append(f"\\cline{{{col_start}-{col_end}}}")
        col_start = col_end + 1

    # Build table body
    lines = ["\\toprule"]
    lines.append(" & ".join(header1) + " \\\\")
    lines.extend(cline_commands)
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")

    for _i, (idx, row) in enumerate(fmt_df.iterrows()):
        # Check if this is BirdMAE and should have blue font for all cells
        idx_lower = idx.lower()
        is_birdmae = "birdmae" in idx_lower or "bird-mae" in idx_lower

        # Format model name
        model_name = f"\\textbf{{{idx}}}"

        # For BirdMAE, make all cells blue (name and all values)
        if is_birdmae:
            # Wrap model name in blue color
            model_name = f"\\textcolor[HTML]{{0000FF}}{{\\textbf{{{idx}}}}}"
            # Wrap all values in blue color
            blue_values = [f"\\textcolor[HTML]{{0000FF}}{{{val}}}" for val in row.values]
            line_parts = [model_name] + blue_values
        else:
            line_parts = [model_name] + list(row.values)

        lines.append(" & ".join(line_parts) + " \\\\")

        # Add separator if this model is in the separators_after list
        # Add separator only for exact matches
        # (avoid substring matches like Perch vs SurfPerch)
        if idx in separators_after:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    body = "\n".join(lines)

    # Build resizebox command if needed
    if resize_width:
        resize_cmd = f"\\resizebox{{{resize_width}}}{{!}}{{"
        resize_close = "}"
    else:
        resize_cmd = ""
        resize_close = ""

    # Complete LaTeX table
    latex_content = f"""
\\begin{{table*}}[t]
\\centering
\\{VISUAL_CONFIG["font_size"]}
\\setlength{{\\tabcolsep}}{{{VISUAL_CONFIG["tabcolsep"]}}}
\\renewcommand{{\\arraystretch}}{{{VISUAL_CONFIG["arraystretch"]}}}
{resize_cmd}
\\begin{{tabular}}{{{colfmt}}}
{body}
\\end{{tabular}}%
{resize_close}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table*}}
"""

    return latex_content, colfmt


def build_latex_table_with_highlighting(
    fmt_df: pd.DataFrame,
    group_info: list[tuple[str, int, list[str]]],
    separators_after: list[str] | None = None,
    caption: str | None = None,
    label: str | None = None,
    resize_width: str | None = None,
) -> tuple[str, str]:
    """Build the complete LaTeX table with EfficientNet row highlighting.

    This is a variant of build_latex_table that highlights EfficientNet model rows
    with light blue color (C6DEE7).

    Parameters
    ----------
    fmt_df : pd.DataFrame
        Formatted DataFrame.
    group_info : list[tuple[str, int, list[str]]]
        List of (group_name, col_count, metrics) tuples.
    separators_after : list[str] | None, optional
        List of model names to add separators after. Defaults to None.
    caption : str | None, optional
        Table caption. Defaults to None.
    label : str | None, optional
        Table label. Defaults to None.
    resize_width : str | None, optional
        Width for resizebox (e.g., '\\textwidth', '0.6\\textwidth',
        or None to disable resizebox). Defaults to None.

    Returns
    -------
    tuple[str, str]
        A tuple containing the LaTeX content and column format string.
    """

    # Use defaults if not provided
    if caption is None:
        caption = TABLE_CAPTION
    if label is None:
        label = TABLE_LABEL
    if separators_after is None:
        separators_after = []
    if resize_width is None:
        resize_width = "\\textwidth"

    # Clean column names and index (also display mapping AUROC/nmi)
    def _display_col(name: str) -> str:
        name = name.replace("_", "-")
        name = name.replace("@", "\\@")
        name = name.replace("R-auc", "R-AUC")
        name = name.replace("C-nmi", "nmi")
        return name

    fmt_df.columns = [_display_col(c) for c in fmt_df.columns]
    fmt_df.index = [i.replace("_", "-") for i in fmt_df.index]

    # Build column format with separators
    sep = VISUAL_CONFIG["group_separator"]
    colfmt = f"l{sep}"  # model column with separator

    for i, (_, col_count, _) in enumerate(group_info):
        colfmt += "c" * col_count
        if i < len(group_info) - 1:  # Add separator except after last group
            colfmt += sep

    # Header row 1: multicolumn for groups
    header1 = [""]  # empty for model column
    for group_name, col_count, _ in group_info:
        formatted_name = f"\\textbf{{{group_name}}}"
        header1.append(f"\\multicolumn{{{col_count}}}{{c}}{{{formatted_name}}}")

    # Header row 2: metric names
    header2 = ["\\textbf{Model}"]
    for _group_name, _, metrics in group_info:
        for m in metrics:
            clean_m = m.replace("@", "\\@")
            header2.append(f"\\textit{{{clean_m}}}")

    # Build partial horizontal lines under group headers
    cline_commands = []
    col_start = 2  # Start after model column
    for _group_name, col_count, _ in group_info:
        col_end = col_start + col_count - 1
        cline_commands.append(f"\\cline{{{col_start}-{col_end}}}")
        col_start = col_end + 1

    # Build table body with EfficientNet highlighting
    lines = ["\\toprule"]
    lines.append(" & ".join(header1) + " \\\\")
    lines.extend(cline_commands)
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")

    for _i, (idx, row) in enumerate(fmt_df.iterrows()):
        # Check if this is BirdMAE and should have blue font for all cells
        idx_lower = idx.lower()
        is_birdmae = "birdmae" in idx_lower or "bird-mae" in idx_lower

        # Format model name
        model_name = f"\\textbf{{{idx}}}"

        # For BirdMAE, make all cells blue (name and all values)
        if is_birdmae:
            # Wrap model name in blue color
            model_name = f"\\textcolor[HTML]{{0000FF}}{{\\textbf{{{idx}}}}}"
            # Wrap all values in blue color
            blue_values = [f"\\textcolor[HTML]{{0000FF}}{{{val}}}" for val in row.values]
            line_parts = [model_name] + blue_values
        else:
            line_parts = [model_name] + list(row.values)

        row_line = " & ".join(line_parts) + " \\\\"

        # Check if this is a model that should be highlighted
        # EfficientNet models start with "EffNet" (case-insensitive)
        # Also highlight SurfPerch, BirdNet, and Perch
        # Note: idx may have decorations (superscripts), so we check if it contains the model name
        should_highlight = (
            idx_lower.startswith("effnet")
            or "surfperch" in idx_lower
            or "birdnet" in idx_lower
            or (
                idx_lower.startswith("perch") and "surfperch" not in idx_lower
            )  # Perch but not SurfPerch
        )

        if should_highlight:
            # Add rowcolor command before the row using HTML color format
            # Requires \usepackage[table]{xcolor} in LaTeX preamble
            row_line = f"\\rowcolor[HTML]{{C6DEE7}}{row_line}"

        lines.append(row_line)

        # Add separator if this model is in the separators_after list
        # Add separator only for exact matches
        # (avoid substring matches like Perch vs SurfPerch)
        if idx in separators_after:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    body = "\n".join(lines)

    # Build resizebox command if needed
    if resize_width:
        resize_cmd = f"\\resizebox{{{resize_width}}}{{!}}{{"
        resize_close = "}"
    else:
        resize_cmd = ""
        resize_close = ""

    # Complete LaTeX table
    # Note: Requires \usepackage[table]{xcolor} in LaTeX preamble
    # Color C6DEE7 is defined inline using HTML format for xcolor package
    latex_content = f"""
\\begin{{table*}}[t]
\\centering
\\{VISUAL_CONFIG["font_size"]}
\\setlength{{\\tabcolsep}}{{{VISUAL_CONFIG["tabcolsep"]}}}
\\renewcommand{{\\arraystretch}}{{{VISUAL_CONFIG["arraystretch"]}}}
{resize_cmd}
\\begin{{tabular}}{{{colfmt}}}
{body}
\\end{{tabular}}%
{resize_close}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table*}}
"""

    return latex_content, colfmt


def generate_sub_table(
    df: pd.DataFrame,
    table_config: dict,
    include_groups: list[str] | None = None,
    comparison_df: pd.DataFrame | None = None,
) -> str | None:
    """Generate a sub-table with specific model filtering and separators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with aggregated results.
    table_config : dict
        Configuration dictionary for the sub-table.
    include_groups : list[str] | None, optional
        Optional list of group names to include.
        If None, includes all groups. Defaults to None.
    comparison_df : pd.DataFrame | None, optional
        Optional DataFrame for side-by-side comparison
        (used for _noesc50 tables). Defaults to None.

    Returns
    -------
    str | None
        LaTeX content string if successful, None otherwise.
    """
    models_list = table_config["models"]
    output_file = table_config["output_file"]
    alt_output_file = table_config.get("alt_output_file")
    caption = table_config["caption"]
    label = table_config["label"]
    show_training_tags = table_config.get("show_training_tags", True)
    append_dagger_to = table_config.get("append_dagger_to", [])

    # Find models that exist in our data (handle separator markers)
    available_models = []
    separators_after = []

    for model in models_list:
        if model == "---SEPARATOR---":
            # Mark the previous model for separator (if any)
            if available_models:
                separators_after.append(available_models[-1])
        else:
            # First try exact match
            if model in df.index:
                available_models.append(model)
            else:
                # Try case-insensitive exact match
                matches = df.index[df.index.astype(str).str.lower() == model.lower()]
                if len(matches) > 0:
                    available_models.append(matches[0])
                else:
                    # Try flexible matching for naming patterns
                    # (escape special regex chars)
                    import re

                    escaped_model = re.escape(model).replace(r"\-", "[-_]?")
                    try:
                        matches = df.index[
                            df.index.astype(str).str.contains(
                                escaped_model, regex=True, case=False, na=False
                            )
                        ]
                        if len(matches) > 0:
                            available_models.append(matches[0])
                        else:
                            print(f"Warning: Model '{model}' not found in data")
                    except Exception as e:
                        print(f"Warning: Model '{model}' not found in data (regex error: {e})")

    if not available_models:
        print(f"No models found for sub-table {output_file}")
        return None

    # Filter dataframe to only include available models (raw names)
    sub_df = df.loc[available_models].copy()

    # If comparison_df is provided, align it with sub_df
    comparison_sub_df = None
    if comparison_df is not None:
        # Filter comparison_df to only include available models
        comparison_sub_df = comparison_df.loc[available_models].copy()
        # Align indices (keep only models present in both)
        common_models = sub_df.index.intersection(comparison_sub_df.index)
        sub_df = sub_df.loc[common_models]
        comparison_sub_df = comparison_sub_df.loc[common_models]

    # Build column structure
    cols, group_info = build_column_structure(
        sub_df, include_groups=include_groups, comparison_df=comparison_sub_df
    )

    # Combine columns from both dataframes if comparison_df is provided
    if comparison_df is not None:
        # Get BEANS Classification columns from both dataframes with suffixes
        metrics = GROUP_METRICS["BEANS Classification"]
        beans_cols_noesc50 = [f"BEANS Classification_{m}_noesc50" for m in metrics]
        beans_cols_wesc50 = [f"BEANS Classification_{m}_wesc50" for m in metrics]
        original_cols = [f"BEANS Classification_{m}" for m in metrics]

        # Get columns from sub_df (without ESC-50) and rename with suffix
        selected_df = sub_df[original_cols].copy()
        selected_df.columns = beans_cols_noesc50

        # Get columns from comparison_sub_df (with ESC-50) and rename with suffix
        comparison_cols = comparison_sub_df[original_cols].copy()
        comparison_cols.columns = beans_cols_wesc50

        # Combine both
        selected_df = pd.concat([selected_df, comparison_cols], axis=1)

        # Reorder columns to match the expected order (noesc50 first, then wesc50)
        final_cols = beans_cols_noesc50 + beans_cols_wesc50
        selected_df = selected_df[final_cols]
    else:
        selected_df = sub_df[cols]

    # Decorate model display names with training tags
    decorated_df, mapping = decorate_model_names_for_table(
        selected_df,
        show_training_tags=show_training_tags,
        append_dagger_to=append_dagger_to,
    )

    # Map separators to decorated names
    decorated_separators = [mapping[m] for m in separators_after if m in mapping]

    # Apply formatting
    formatted_df = apply_bold_formatting(decorated_df)

    # Use appropriate resize width
    # For _noesc50 versions with comparison, we have 6 columns (3+3), so use wider width
    if comparison_df is not None:
        resize_width = "0.85\\textwidth"  # Wider for side-by-side comparison
    elif include_groups == ["BEANS Classification"]:
        resize_width = "0.6\\textwidth"
    else:
        resize_width = "\\textwidth"

    # Generate LaTeX table
    latex_content, colfmt = build_latex_table(
        formatted_df,
        group_info,
        separators_after=decorated_separators,
        caption=caption,
        label=label,
        resize_width=resize_width,
    )

    # Save to file(s)
    Path(output_file).write_text(latex_content)
    if alt_output_file:
        Path(alt_output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(alt_output_file).write_text(latex_content)

    print(f"✅ Generated sub-table: {output_file}")
    if alt_output_file:
        print(f"   (also saved to {alt_output_file})")
    print(f"   Models: {len(available_models)} ({', '.join(available_models)})")
    print(f"   Separators after: {decorated_separators}")

    return latex_content


def generate_main_table(
    df: pd.DataFrame,
    output_file: str,
    table_label_suffix: str = "",
    group_datasets: dict | None = None,
    include_groups: list[str] | None = None,
    comparison_df: pd.DataFrame | None = None,
) -> None:
    """Generate the main aggregate table.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with aggregated results.
    output_file : str
        Output file path.
    table_label_suffix : str, optional
        Suffix to add to table label. Defaults to "".
    group_datasets : dict | None, optional
        Group datasets dictionary (for reporting). Defaults to None.
    include_groups : list[str] | None, optional
        Optional list of group names to include.
        If None, includes all groups. Defaults to None.
    comparison_df : pd.DataFrame | None, optional
        Optional DataFrame for side-by-side comparison
        (used for _noesc50 tables). Defaults to None.
    """
    if group_datasets is None:
        group_datasets = CORRECTED_GROUP_DATASETS

    # Order models: existing/pretrained on top, new models below
    df_ordered, last_existing_raw = reorder_models_with_separator(df)

    # If comparison_df is provided, align it with df_ordered
    # and combine BEANS Classification columns
    if comparison_df is not None:
        # Reorder comparison_df to match df_ordered
        comparison_ordered, _ = reorder_models_with_separator(comparison_df)
        # Align indices (keep only models present in both)
        common_models = df_ordered.index.intersection(comparison_ordered.index)
        df_ordered = df_ordered.loc[common_models]
        comparison_ordered = comparison_ordered.loc[common_models]

    cols, group_info = build_column_structure(
        df_ordered,
        include_groups=include_groups,
        comparison_df=comparison_ordered if comparison_df is not None else None,
    )

    # Combine columns from both dataframes if comparison_df is provided
    if comparison_df is not None:
        # Get BEANS Classification columns from both dataframes with suffixes
        metrics = GROUP_METRICS["BEANS Classification"]
        beans_cols_noesc50 = [f"BEANS Classification_{m}_noesc50" for m in metrics]
        beans_cols_wesc50 = [f"BEANS Classification_{m}_wesc50" for m in metrics]
        original_cols = [f"BEANS Classification_{m}" for m in metrics]

        # Get columns from df (without ESC-50) and rename with suffix
        selected_df = df_ordered[original_cols].copy()
        selected_df.columns = beans_cols_noesc50

        # Get columns from comparison_df (with ESC-50) and rename with suffix
        comparison_cols = comparison_ordered[original_cols].copy()
        comparison_cols.columns = beans_cols_wesc50

        # Combine both
        selected_df = pd.concat([selected_df, comparison_cols], axis=1)

        # Reorder columns to match the expected order (noesc50 first, then wesc50)
        final_cols = beans_cols_noesc50 + beans_cols_wesc50
        selected_df = selected_df[final_cols]
    else:
        selected_df = df_ordered[cols]

    print("✨ Applying formatting...")
    # Decorate names with training tags and rename NatureBEATs
    decorated_df, name_map = decorate_model_names_for_table(selected_df)
    formatted_df = apply_bold_formatting(decorated_df)

    print("📝 Building main LaTeX table...")
    # Add separator after the last existing/pretrained model
    if name_map:
        main_separators = [name_map.get(last_existing_raw, next(iter(name_map.values())))]
    else:
        main_separators = []

    # Update caption and label if needed
    caption = TABLE_CAPTION
    label = TABLE_LABEL
    if table_label_suffix:
        label = label + table_label_suffix

    # Use appropriate resize width
    # For _noesc50 versions with comparison, we have 6 columns (3+3), so use wider width
    if comparison_df is not None:
        resize_width = "0.85\\textwidth"  # Wider for side-by-side comparison
    elif include_groups == ["BEANS Classification"] or (
        table_label_suffix and "noesc50" in table_label_suffix
    ):
        resize_width = "0.6\\textwidth"
    else:
        resize_width = "\\textwidth"

    latex_content, colfmt = build_latex_table(
        formatted_df,
        group_info,
        separators_after=main_separators,
        caption=caption,
        label=label,
        resize_width=resize_width,
    )

    print(f"💾 Saving main table to {output_file}...")
    Path(output_file).write_text(latex_content)

    print(f"✅ Success! Generated main table: {output_file}")
    print(f"📏 Table dimensions: {len(formatted_df)} models × {len(cols)} metrics")
    print(f"🎨 Column format: {colfmt}")
    print(f"📋 Groups: {[name for name, _, _ in group_info]}")

    # Count datasets in BEANS Classification
    beans_class_datasets = len(group_datasets.get("BEANS Classification", []))
    print(f"🔧 BEANS Classification: {beans_class_datasets} datasets")
    print("🐦 BirdSet: 7 datasets (including SNE, UHH)")


def generate_main_table_highlighted(
    df: pd.DataFrame,
    output_file: str,
    table_label_suffix: str = "",
    group_datasets: dict | None = None,
    include_groups: list[str] | None = None,
    comparison_df: pd.DataFrame | None = None,
) -> None:
    """Generate the main aggregate table with EfficientNet highlighting.

    This is a variant of generate_main_table that highlights EfficientNet model rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with aggregated results.
    output_file : str
        Output file path.
    table_label_suffix : str, optional
        Suffix to add to table label. Defaults to "".
    group_datasets : dict | None, optional
        Group datasets dictionary (for reporting). Defaults to None.
    include_groups : list[str] | None, optional
        Optional list of group names to include.
        If None, includes all groups. Defaults to None.
    comparison_df : pd.DataFrame | None, optional
        Optional DataFrame for side-by-side comparison
        (used for _noesc50 tables). Defaults to None.
    """
    if group_datasets is None:
        group_datasets = CORRECTED_GROUP_DATASETS

    # Order models: existing/pretrained on top, new models below
    df_ordered, last_existing_raw = reorder_models_with_separator(df)

    # If comparison_df is provided, align it with df_ordered
    # and combine BEANS Classification columns
    if comparison_df is not None:
        # Reorder comparison_df to match df_ordered
        comparison_ordered, _ = reorder_models_with_separator(comparison_df)
        # Align indices (keep only models present in both)
        common_models = df_ordered.index.intersection(comparison_ordered.index)
        df_ordered = df_ordered.loc[common_models]
        comparison_ordered = comparison_ordered.loc[common_models]

    cols, group_info = build_column_structure(
        df_ordered,
        include_groups=include_groups,
        comparison_df=comparison_ordered if comparison_df is not None else None,
    )

    # Combine columns from both dataframes if comparison_df is provided
    if comparison_df is not None:
        # Get BEANS Classification columns from both dataframes with suffixes
        metrics = GROUP_METRICS["BEANS Classification"]
        beans_cols_noesc50 = [f"BEANS Classification_{m}_noesc50" for m in metrics]
        beans_cols_wesc50 = [f"BEANS Classification_{m}_wesc50" for m in metrics]
        original_cols = [f"BEANS Classification_{m}" for m in metrics]

        # Get columns from df (without ESC-50) and rename with suffix
        selected_df = df_ordered[original_cols].copy()
        selected_df.columns = beans_cols_noesc50

        # Get columns from comparison_df (with ESC-50) and rename with suffix
        comparison_cols = comparison_ordered[original_cols].copy()
        comparison_cols.columns = beans_cols_wesc50

        # Combine both
        selected_df = pd.concat([selected_df, comparison_cols], axis=1)

        # Reorder columns to match the expected order (noesc50 first, then wesc50)
        final_cols = beans_cols_noesc50 + beans_cols_wesc50
        selected_df = selected_df[final_cols]
    else:
        selected_df = df_ordered[cols]

    print("✨ Applying formatting...")
    # Decorate names with training tags and rename NatureBEATs
    decorated_df, name_map = decorate_model_names_for_table(selected_df)
    formatted_df = apply_bold_formatting(decorated_df)

    print("📝 Building main LaTeX table with EfficientNet highlighting...")
    # Add separator after the last existing/pretrained model
    if name_map:
        main_separators = [name_map.get(last_existing_raw, next(iter(name_map.values())))]
    else:
        main_separators = []

    # Update caption and label if needed
    # For highlighted table, add note about EfficientNet shading
    caption = TABLE_CAPTION + " EfficientNet models are shaded."
    label = TABLE_LABEL
    if table_label_suffix:
        label = label + table_label_suffix

    # Use appropriate resize width
    # For _noesc50 versions with comparison, we have 6 columns (3+3), so use wider width
    if comparison_df is not None:
        resize_width = "0.85\\textwidth"  # Wider for side-by-side comparison
    elif include_groups == ["BEANS Classification"] or (
        table_label_suffix and "noesc50" in table_label_suffix
    ):
        resize_width = "0.6\\textwidth"
    else:
        resize_width = "\\textwidth"

    latex_content, colfmt = build_latex_table_with_highlighting(
        formatted_df,
        group_info,
        separators_after=main_separators,
        caption=caption,
        label=label,
        resize_width=resize_width,
    )

    print(f"💾 Saving highlighted table to {output_file}...")
    Path(output_file).write_text(latex_content)

    print(f"✅ Success! Generated highlighted table: {output_file}")
    print(f"📏 Table dimensions: {len(formatted_df)} models × {len(cols)} metrics")
    print(f"🎨 Column format: {colfmt}")
    print(f"📋 Groups: {[name for name, _, _ in group_info]}")
    print("🎨 EfficientNet, Perch, SurfPerch, and BirdNet rows highlighted with color C6DEE7")
    print("🎨 BirdMAE row displayed in blue font (color 0000FF) with name 'Bird-MAE-Huge'")

    # Count datasets in BEANS Classification
    beans_class_datasets = len(group_datasets.get("BEANS Classification", []))
    print(f"🔧 BEANS Classification: {beans_class_datasets} datasets")
    print("🐦 BirdSet: 7 datasets (including SNE, UHH)")


def main() -> None:
    """Main function to generate the enhanced table and sub-tables."""
    print("🔄 Loading and processing data...")
    df = load_and_process_data()

    # Generate main table with separator after Perch
    print("\n📊 Generating main table...")
    generate_main_table(df, OUTPUT_FILE, group_datasets=CORRECTED_GROUP_DATASETS)

    # Generate highlighted version with EfficientNet rows highlighted
    print("\n📊 Generating main table (highlighted version with EfficientNet rows)...")
    generate_main_table_highlighted(
        df, OUTPUT_FILE_HIGHLIGHTED, group_datasets=CORRECTED_GROUP_DATASETS
    )

    # Generate _noesc50 version (BEANS Classification without ESC-50
    # vs with ESC-50 side-by-side)
    print(
        "\n📊 Generating main table (BEANS Classification comparison: w/o ESC-50 vs w/ ESC-50)..."
    )
    group_datasets_noesc50 = get_group_datasets_noesc50()
    df_noesc50 = load_and_process_data(group_datasets=group_datasets_noesc50)
    # Pass the regular df (with ESC-50) as comparison_df for side-by-side comparison
    generate_main_table(
        df_noesc50,
        OUTPUT_FILE_NOESC50,
        table_label_suffix="-noesc50",
        group_datasets=group_datasets_noesc50,
        include_groups=["BEANS Classification"],
        comparison_df=df,
    )

    # Generate sub-tables (original versions)
    print("\n🔄 Generating sub-tables...")
    for table_name, table_config in SUB_TABLES.items():
        print(f"\n📊 Processing {table_name}...")
        try:
            generate_sub_table(df, table_config)
        except Exception as e:
            print(f"❌ Error generating {table_name}: {e}")

    # Generate _noesc50 versions of sub-tables
    print("\n🔄 Generating sub-tables (no ESC-50)...")
    for table_name, table_config in SUB_TABLES.items():
        print(f"\n📊 Processing {table_name}_noesc50...")
        try:
            # Create a copy of the config with updated file names and labels
            noesc50_config = table_config.copy()
            # Update output file names
            if "output_file" in noesc50_config:
                base_name = Path(noesc50_config["output_file"]).stem
                noesc50_config["output_file"] = str(
                    Path(noesc50_config["output_file"]).parent / f"{base_name}_noesc50.tex"
                )
            if "alt_output_file" in noesc50_config:
                base_name = Path(noesc50_config["alt_output_file"]).stem
                noesc50_config["alt_output_file"] = str(
                    Path(noesc50_config["alt_output_file"]).parent / f"{base_name}_noesc50.tex"
                )
            # Update label
            noesc50_config["label"] = table_config["label"] + "-noesc50"
            # Use df_noesc50 for the _noesc50 version, only include BEANS Classification
            # Pass the regular df (with ESC-50) as comparison_df
            # for side-by-side comparison
            generate_sub_table(
                df_noesc50,
                noesc50_config,
                include_groups=["BEANS Classification"],
                comparison_df=df,
            )
        except Exception as e:
            print(f"❌ Error generating {table_name}_noesc50: {e}")

    print("\n🎉 All tables generated successfully!")
    print(f"📁 Main table: {OUTPUT_FILE}")
    print(f"📁 Main table (no ESC-50): {OUTPUT_FILE_NOESC50}")
    print(f"📁 Main table (highlighted): {OUTPUT_FILE_HIGHLIGHTED}")
    for table_name, config in SUB_TABLES.items():
        print(f"📁 {table_name}: {config['output_file']}")
        if "alt_output_file" in config:
            print(f"📁 {table_name} (alt): {config['alt_output_file']}")
        # Show _noesc50 versions
        base_name = Path(config["output_file"]).stem
        noesc50_file = str(Path(config["output_file"]).parent / f"{base_name}_noesc50.tex")
        print(f"📁 {table_name}_noesc50: {noesc50_file}")


if __name__ == "__main__":
    main()
