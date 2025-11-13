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

import pandas as pd
import ast
import numpy as np
import sys
from pathlib import Path

# Add scripts/analysis to path for imports
sys.path.append('scripts/analysis')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input Excel file path
EXCEL_PATH = 'static/results/Representation Learning Results (35).xlsx'

# Output LaTeX file
OUTPUT_FILE = 'latex_arxiv/aggregate_v0.tex'

# Define which metrics to include for each benchmark group
# Keep internal metric keys as in the data (R-auc, C-nmi), and map to AUROC/nmi only for display
GROUP_METRICS = {
    "BEANS Classification": ["Probe", "R-auc", "C-nmi"],
    "BEANS Detection": ["Probe", "R-auc"], 
    "BirdSet": ["Probe", "R-auc"],
    "Individual ID": ["Probe", "R-auc"],
    "Vocal Repertoire": ["R-auc", "C-nmi"]
}

# Override group datasets to ensure correct groupings
CORRECTED_GROUP_DATASETS = {
    "BEANS Classification": [
        "Watkins", "CBI", "HBDB", "BATS", "Dogs", "ESC-50"
    ],
    "BEANS Detection": [
        "enabirds", "rfcx", "hiceas", "gibbons", "dcase"
    ],
    "BirdSet": [
        "POW", "PER", "NES", "NBP", "HSN", "SNE", "UHH"
    ],
    "Individual ID": [
        "chiffchaff-cross", "littleowls-cross", "pipit-cross", "macaques"
    ],
    "Vocal Repertoire": [
        "zebrafinch-je-call", "Giant_Otters", "Bengalese_Finch", "SRKW_Orca"
    ],
}

# Models to exclude from the table
EXCLUDE_MODELS = [
    'BirdMAE',
    'CLAP',
    'EffNetB0-bio-wabad',
    'EffNetB0-bio-soundscape',
    'EffNetB0-no-whales',
    'EffNetB0-no-birds',
    'EffNetB0-birds',
]

# Shared caption suffix used in captions to maintain consistency
METRIC_CAPTION_SUFFIX = (
    "We report ROC AUC for retrieval, accuracy for probing on BEANS classification and Individual ID, "
    "mean-average precision for probe on BEANS Detection and BirdSet. We report the mean of each metric over datasets per benchmark. $^{\dagger}$BirdNet results on BirdSet are excluded following the authors \citep{rauchbirdset} due to data leakage"
)

# Define sub-tables with model groupings and separators
SUB_TABLES = {
    'efficientnet_analysis': {
        'models': [
            # Pretrained models first
            'Perch',
            'SurfPerch',
            'BirdNet',
            # Separator after pretrained models
            '---SEPARATOR---',
            # Our EffNet models (AudioSet, bio, all)
            'EffNetB0-AudioSet',
            'EffNetB0-bio', 
            'EffNetB0-all'
        ],
        'output_file': 'latex/efficientnet_analysis.tex',
        'alt_output_file': 'latex_arxiv/efficientnet_analysis.tex',
        'caption': 'Comparison of EfficientNet models trained on mixes of AudioSet and large-scale bioacoustic data. ' + METRIC_CAPTION_SUFFIX,
        'show_training_tags': False,
        'append_dagger_to': ['BirdNet'],
        'label': 'tab:efficientnet-analysis'
    },
    'ssl_models_analysis': {
        'models': [
            # Our EAT models only (remove sl- post-trained models)
                        # Pretrained SSL models
            'BEATS (SFT)',
            'BEATS (pretrained)',
            'Bird-AVES-biox-base',
            # Exclude NatureLM-audio as requested
            '---SEPARATOR---',
            'EAT-AS',
            'EAT-bio', 
            'EAT-all',
            # Separator after our models
        ],
        'output_file': 'latex/ssl_models_analysis.tex', 
        'caption': 'SSL model analysis: our trained models vs. pretrained SSL baselines.',
        'label': 'tab:ssl-analysis'
    }
}

# Visual formatting options
VISUAL_CONFIG = {
    'font_size': 'footnotesize',      # tiny, scriptsize, footnotesize, small
    'tabcolsep': '3pt',               # column spacing
    'arraystretch': '1.2',            # row height multiplier
    'group_separator': '||',          # separator between groups (| or ||)
    'decimal_places': 3,              # number of decimal places
}

TABLE_CAPTION = (
    "Aggregate results across bioacoustic benchmarks and tasks (best per metric in bold). "
    + METRIC_CAPTION_SUFFIX +
    "Model labels carry training tags: \\textsuperscript{SSL} self-supervised, \\textsuperscript{SL} supervised, "
    "\\textsuperscript{SL-SSL} supervised fine-tuning after SSL pretraining. Models above the midrule are existing/pretrained checkpoints; below are new models from this work."
)
TABLE_LABEL = "tab:aggregate_results"

# =============================================================================
# SCRIPT
# =============================================================================

def extract_groups_and_metrics():
    """Extract metrics from visualize_excel.py, but use our corrected group datasets"""
    code = Path('scripts/analysis/visualize_excel.py').read_text()
    main = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name=='main'][0]
    
    metrics = None
    
    for node in main.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    if t.id == 'metrics':
                        metrics = ast.literal_eval(node.value)
    
    # Use our corrected group datasets instead of the original
    return CORRECTED_GROUP_DATASETS, metrics

def validate_dataset_coverage(df: pd.DataFrame, group_datasets: dict, metrics: list):
    """Validate that all expected datasets exist in the data and report missing ones."""
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
    
    print(f"\n📊 Summary:")
    print(f"  Found datasets: {len(all_found)}")
    print(f"  Missing datasets: {len(all_missing)}")
    
    if all_missing:
        print(f"\n❌ Missing datasets:")
        for group, dataset in all_missing:
            print(f"  {group}: {dataset}")
    
    return all_missing, all_found

def compute_group_averages_with_prioritization(
    df: pd.DataFrame,
    group_datasets: dict,
    metrics: list,
) -> pd.DataFrame:
    """Compute per-model averages with special handling for Individual ID cross-retrieval and BirdNet exclusion on BirdSet."""
    result = pd.DataFrame(index=df.index)
    
    for group_name, datasets in group_datasets.items():
        print(f"\nProcessing {group_name} with datasets: {datasets}")
        
        for metric in metrics:
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
                        print(f"  ⚠️  Found {total_nans}/{total_cells} empty cells, excluding from average")
                        for col, nan_count in nan_counts.items():
                            if nan_count > 0:
                                print(f"     {col}: {nan_count} empty")
                    
                    avg = combined_df.mean(axis=1, skipna=True)
                    result[f"{group_name}_{metric}"] = avg
                    print(f"  Individual ID R-auc: Combined {len(prioritized_values)} datasets")
                
            else:
                # Standard aggregation for all other cases
                keys = [
                    f"{ds}_{metric}" for ds in datasets if f"{ds}_{metric}" in df.columns
                ]
                missing_keys = [f"{ds}_{metric}" for ds in datasets if f"{ds}_{metric}" not in df.columns]
                
                if missing_keys:
                    print(f"  ❌ Missing columns for {group_name} {metric}: {missing_keys}")
                
                if keys:
                    values = df[keys].apply(pd.to_numeric, errors="coerce")
                    
                    # Special handling: exclude BirdNet results on BirdSet datasets
                    if group_name == "BirdSet":
                        # Handle potential case variations or exact name matching
                        birdnet_rows = values.index[values.index.str.contains("BirdNet", case=False, na=False)]
                        if len(birdnet_rows) > 0:
                            # Store the exclusion info before computing average
                            birdnet_exclusion = list(birdnet_rows)
                            values.loc[birdnet_rows] = np.nan
                            print(f"  🚫 Excluding BirdNet results for BirdSet datasets: {birdnet_exclusion}")
                        else:
                            print(f"  ℹ️  BirdNet not found in index for BirdSet exclusion")
                    
                    # Check for NaN values before averaging
                    nan_counts = values.isna().sum()
                    total_cells = len(values) * len(values.columns)
                    total_nans = nan_counts.sum()
                    
                    if total_nans > 0:
                        print(f"  ⚠️  Found {total_nans}/{total_cells} empty cells, excluding from average")
                        # Report which models have missing data
                        models_with_nans = values.isna().any(axis=1)
                        if models_with_nans.any():
                            nan_models = values.index[models_with_nans].tolist()
                            print(f"     Models with missing data: {nan_models}")
                    
                    avg = values.mean(axis=1, skipna=True)
                    result[f"{group_name}_{metric}"] = avg
                    print(f"  ✅ {group_name} {metric}: Found {len(keys)}/{len(datasets)} datasets - {keys}")
                else:
                    print(f"  ❌ No data found for {group_name} {metric}")
    
    return result

def load_and_process_data():
    """Load Excel data and compute group averages with prioritization"""
    from visualize_excel import load_excel
    
    group_datasets, metrics = extract_groups_and_metrics()
    
    # Load raw data
    raw = load_excel(EXCEL_PATH)
    
    print(f"📋 Raw data shape: {raw.shape}")
    print(f"🏷️  Available columns: {len(raw.columns)} total")
    
    # Show corrected groupings
    print(f"🔧 Corrected BEANS Classification: {CORRECTED_GROUP_DATASETS['BEANS Classification']}")
    print(f"🔧 BirdSet: {CORRECTED_GROUP_DATASETS['BirdSet']}")
    
    # Validate dataset coverage before processing
    missing_datasets, found_datasets = validate_dataset_coverage(raw, group_datasets, metrics)
    
    # Compute group averages with special Individual ID handling
    print("\n🔄 Computing group averages with cross-retrieval prioritization...")
    full = compute_group_averages_with_prioritization(raw, group_datasets, metrics)
    
    # Remove clustering ARI columns, keep NMI
    full = full.drop(columns=[c for c in full.columns if c.endswith('C-ari')], errors='ignore')
    full = full.round(VISUAL_CONFIG['decimal_places'])
    
    # Filter out specified models
    mask = ~full.index.str.contains('|'.join(EXCLUDE_MODELS))
    full = full[mask]
    
    return full

def build_column_structure(df):
    """Build column structure based on GROUP_METRICS configuration"""
    cols = []
    group_info = []
    
    for group_name, metrics in GROUP_METRICS.items():
        g_cols = []
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
    """Apply explicit display-name overrides (e.g., NatureBEATs)."""
    rename_map = {
        "BEATS-NatureLM-audio": "NatureBEATs",
    }
    return rename_map.get(raw_name, raw_name)

def decorate_model_names_for_table(df: pd.DataFrame, *, show_training_tags: bool = True, append_dagger_to: list | None = None) -> tuple[pd.DataFrame, dict]:
    """Return a copy of df where the index is replaced with display names including training tags.

    Also return a mapping from raw name -> decorated display name for downstream use
    (e.g., to place separators after a specific model).
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

    Returns the reordered frame and the raw-name of the last 'existing' model to place a midrule after.
    """
    raw_models = df.index.tolist()

    preferred_existing = [
        "BEATS (SFT)",
        "BEATS (pretrained)",
        "EAT-base (pretrained)",
        "EAT-base (SFT)",
        "Bird-AVES-biox-base",
        "BEATS-NatureLM-audio",  # will be shown as NatureBEATs
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
        l = name.lower()
        if l in {"perch", "surfperch", "birdnet"}:
            return True
        if l.startswith("beats") or l.startswith("bird-aves"):
            return True
        if l.startswith("eat-base"):
            return True
        if l == "beats-naturelm-audio":
            return True
        return False

    ordered = []

    # Add preferred existing in order
    for m in preferred_existing:
        if m in raw_models:
            ordered.append(m)

    # Add any other existing not yet included
    for m in raw_models:
        if m not in ordered and is_existing(m):
            ordered.append(m)

    # Remember the last existing model for the separator; if none, place after first group by default
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

def apply_bold_formatting(df):
    """Apply bold formatting to maximum values in each column, with special N/A handling for BirdNet on BirdSet"""
    fmt = df.copy().astype(str)
    
    for col in df.columns:
        maxv = df[col].max(skipna=True)
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                # Special case: BirdNet on BirdSet datasets gets "N/A"
                if "BirdNet" in idx and "BirdSet" in col:
                    fmt.at[idx, col] = 'N/A'
                else:
                    fmt.at[idx, col] = '---'
            else:
                if abs(val - maxv) < 1e-6:
                    fmt.at[idx, col] = f"\\textbf{{{val:.{VISUAL_CONFIG['decimal_places']}f}}}"
                else:
                    fmt.at[idx, col] = f"{val:.{VISUAL_CONFIG['decimal_places']}f}"
    
    return fmt

def build_latex_table(fmt_df, group_info, separators_after=None, caption=None, label=None):
    """Build the complete LaTeX table with optional separators"""
    
    # Use defaults if not provided
    if caption is None:
        caption = TABLE_CAPTION
    if label is None:
        label = TABLE_LABEL
    if separators_after is None:
        separators_after = []
    
    # Clean column names and index (also display mapping AUROC/nmi)
    def _display_col(name: str) -> str:
        name = name.replace('_', '-')
        name = name.replace('@', '\\@')
        name = name.replace('R-auc', 'R-AUC')
        name = name.replace('C-nmi', 'nmi')
        return name

    fmt_df.columns = [_display_col(c) for c in fmt_df.columns]
    fmt_df.index = [i.replace('_', '-') for i in fmt_df.index]
    
    # Build column format with separators
    sep = VISUAL_CONFIG['group_separator']
    colfmt = f'l{sep}'  # model column with separator
    
    for i, (_, col_count, _) in enumerate(group_info):
        colfmt += 'c' * col_count
        if i < len(group_info) - 1:  # Add separator except after last group
            colfmt += sep
    
    # Header row 1: multicolumn for groups
    header1 = ['']  # empty for model column
    for group_name, col_count, _ in group_info:
        formatted_name = f"\\textbf{{{group_name}}}"
        header1.append(f"\\multicolumn{{{col_count}}}{{c}}{{{formatted_name}}}")
    
    # Header row 2: metric names
    header2 = ['\\textbf{Model}']
    for group_name, _, metrics in group_info:
        for m in metrics:
            clean_m = m.replace('@', '\\@')
            header2.append(f"\\textit{{{clean_m}}}")
    
    # Build partial horizontal lines under group headers
    cline_commands = []
    col_start = 2  # Start after model column
    for group_name, col_count, _ in group_info:
        col_end = col_start + col_count - 1
        cline_commands.append(f"\\cline{{{col_start}-{col_end}}}")
        col_start = col_end + 1
    
    # Build table body
    lines = ['\\toprule']
    lines.append(' & '.join(header1) + ' \\\\')
    lines.extend(cline_commands)
    lines.append(' & '.join(header2) + ' \\\\')
    lines.append('\\midrule')
    
    for i, (idx, row) in enumerate(fmt_df.iterrows()):
        line_parts = [f"\\textbf{{{idx}}}"] + list(row.values)
        lines.append(' & '.join(line_parts) + ' \\\\')
        
        # Add separator if this model is in the separators_after list
        # Add separator only for exact matches (avoid substring matches like Perch vs SurfPerch)
        if idx in separators_after:
            lines.append('\\midrule')
    
    lines.append('\\bottomrule')
    body = '\n'.join(lines)
    
    # Complete LaTeX table
    latex_content = f"""
\\begin{{table*}}[t]
\\centering
\\{VISUAL_CONFIG['font_size']}
\\setlength{{\\tabcolsep}}{{{VISUAL_CONFIG['tabcolsep']}}}
\\renewcommand{{\\arraystretch}}{{{VISUAL_CONFIG['arraystretch']}}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{colfmt}}}
{body}
\\end{{tabular}}%
}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table*}}
"""
    
    return latex_content, colfmt

def generate_sub_table(df, table_config):
    """Generate a sub-table with specific model filtering and separators"""
    models_list = table_config['models']
    output_file = table_config['output_file']
    alt_output_file = table_config.get('alt_output_file')
    caption = table_config['caption']
    label = table_config['label']
    show_training_tags = table_config.get('show_training_tags', True)
    append_dagger_to = table_config.get('append_dagger_to', [])
    
    # Find models that exist in our data (handle separator markers)
    available_models = []
    separators_after = []
    
    for model in models_list:
        if model == '---SEPARATOR---':
            # Mark the previous model for separator (if any)
            if available_models:
                separators_after.append(available_models[-1])
        else:
            # First try exact match
            if model in df.index:
                available_models.append(model)
            else:
                # Try case-insensitive exact match
                matches = df.index[df.index.str.lower() == model.lower()]
                if len(matches) > 0:
                    available_models.append(matches[0])
                else:
                    # Try flexible matching for naming patterns (escape special regex chars)
                    import re
                    escaped_model = re.escape(model).replace(r'\-', '[-_]?')
                    try:
                        matches = df.index[df.index.str.contains(escaped_model, regex=True, case=False)]
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
    
    # Build column structure
    cols, group_info = build_column_structure(sub_df)
    selected_df = sub_df[cols]

    # Decorate model display names with training tags
    decorated_df, mapping = decorate_model_names_for_table(
        selected_df, show_training_tags=show_training_tags, append_dagger_to=append_dagger_to
    )

    # Map separators to decorated names
    decorated_separators = [mapping[m] for m in separators_after if m in mapping]

    # Apply formatting
    formatted_df = apply_bold_formatting(decorated_df)
    
    # Generate LaTeX table
    latex_content, colfmt = build_latex_table(
        formatted_df,
        group_info,
        separators_after=decorated_separators,
        caption=caption,
        label=label
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

def main():
    """Main function to generate the enhanced table and sub-tables"""
    print("🔄 Loading and processing data...")
    df = load_and_process_data()
    
    # Generate main table with separator after Perch
    print("\n📊 Generating main table...")
    # Order models: existing/pretrained on top, new models below
    df_ordered, last_existing_raw = reorder_models_with_separator(df)

    cols, group_info = build_column_structure(df_ordered)
    selected_df = df_ordered[cols]
    
    print("✨ Applying formatting...")
    # Decorate names with training tags and rename NatureBEATs
    decorated_df, name_map = decorate_model_names_for_table(selected_df)
    formatted_df = apply_bold_formatting(decorated_df)
    
    print("📝 Building main LaTeX table...")
    # Add separator after the last existing/pretrained model
    main_separators = [name_map.get(last_existing_raw, next(iter(name_map.values())))]
    latex_content, colfmt = build_latex_table(
        formatted_df, 
        group_info, 
        separators_after=main_separators
    )
    
    print("💾 Saving main table...")
    Path(OUTPUT_FILE).write_text(latex_content)
    
    print(f"✅ Success! Generated main table: {OUTPUT_FILE}")
    print(f"📏 Table dimensions: {len(formatted_df)} models × {len(cols)} metrics")
    print(f"🎨 Column format: {colfmt}")
    print(f"📋 Groups: {[name for name, _, _ in group_info]}")
    print("🔧 BEANS Classification: 6 datasets")
    print("🐦 BirdSet: 7 datasets (including SNE, UHH)")
    
    # Generate sub-tables
    print("\n🔄 Generating sub-tables...")
    for table_name, table_config in SUB_TABLES.items():
        print(f"\n📊 Processing {table_name}...")
        try:
            generate_sub_table(df, table_config)
        except Exception as e:
            print(f"❌ Error generating {table_name}: {e}")
    
    print(f"\n🎉 All tables generated successfully!")
    print(f"📁 Main table: {OUTPUT_FILE}")
    for table_name, config in SUB_TABLES.items():
        print(f"📁 {table_name}: {config['output_file']}")

if __name__ == "__main__":
    main()
