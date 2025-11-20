import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def _detect_benchmark(df: pd.DataFrame) -> str:
    """
    Detect the benchmark contained in the input CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a `benchmark` column.

    Returns
    -------
    str
        Detected benchmark name: "beans" or "birdset".

    Raises
    ------
    ValueError
        If `benchmark` column is missing, contains multiple values, or is empty.
    """
    if "benchmark" not in df.columns:
        raise ValueError("Input CSV is missing required 'benchmark' column")
    unique_benchmarks = df["benchmark"].dropna().unique().tolist()
    if len(unique_benchmarks) == 0:
        raise ValueError("No benchmark values found in input CSV")
    if len(unique_benchmarks) > 1:
        raise ValueError(f"Multiple benchmarks found: {unique_benchmarks}. Please filter first.")
    bench = str(unique_benchmarks[0]).strip()
    if bench not in {"beans", "birdset"}:
        raise ValueError(f"Unsupported benchmark '{bench}'")
    return bench


def _validate_input_columns(df: pd.DataFrame) -> None:
    """
    Validate required columns exist in the long-form CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = {
        "dataset_name",
        "metric",
        "base_model",
        "benchmark",
        "probe_type",
        "layers",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")


def _extract_probe_info_from_base_model(base_model: str) -> Tuple[str, str]:
    """
    Extract probe_type and layers from base_model name.

    Parameters
    ----------
    base_model : str
        Base model name like "efficientnet_animalspeak_audioset_attention_last"

    Returns
    -------
    tuple of str
        (probe_type, layers) where probe_type is "attention" or "linear"
        and layers is "last_layer" or "all"
    """
    # Remove "weighted_" prefix if present
    model = base_model.replace("weighted_", "")

    # Extract probe type and layer from the end
    if "_attention_last" in model:
        return "attention", "last_layer"
    elif "_attention_all" in model:
        return "attention", "all"
    elif "_linear_last" in model:
        return "linear", "last_layer"
    elif "_linear_all" in model:
        return "linear", "all"
    else:
        # Fallback: try to extract from the end
        parts = model.split("_")
        if len(parts) >= 2:
            layer = parts[-1]
            probe_type = parts[-2]
            return probe_type, layer
        return "unknown", "unknown"


def _pivot_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-form metrics to wide form with datasets as columns.

    Rows are indexed by `base_model`. If the input has multiple rows for the
    same `(base_model, dataset_name)`, the mean of `metric` is used.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form dataframe filtered to a single benchmark.

    Returns
    -------
    pd.DataFrame
        Wide-form dataframe with index `base_model` and columns as datasets.
    """
    grouped = (
        df[["base_model", "dataset_name", "metric"]]
        .groupby(["base_model", "dataset_name"], as_index=False)
        .agg({"metric": "mean"})
    )
    wide = grouped.pivot(index="base_model", columns="dataset_name", values="metric")
    wide = wide.sort_index(axis=0).sort_index(axis=1)
    # Ensure float dtype
    wide = wide.astype(float)
    return wide


def _create_base_model_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all 4 combinations of (probe_type, layers) for each base model.

    For each base model, create 4 new rows representing:
    - (attention, last_layer)
    - (attention, all)
    - (linear, last_layer)
    - (linear, all)

    Each combination gets a unique base_model name that includes the
    probe_type and layers.
    If a combination already exists, use the existing data.
    If missing, create placeholder rows with NaN metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form dataframe with columns: base_model, dataset_name, metric,
        probe_type, layers

    Returns
    -------
    pd.DataFrame
        Dataframe with all 4 combinations for each base model
    """
    # Get unique base models and datasets
    base_models = df["base_model"].unique()
    datasets = df["dataset_name"].unique()

    # Expected combinations
    expected_combinations = [
        ("weighted_attention", "last_layer"),
        ("weighted_attention", "all"),
        ("weighted_linear", "last_layer"),
        ("weighted_linear", "all"),
    ]

    # Create all possible combinations
    all_rows = []
    for base_model in base_models:
        # Extract the base model name without the probe type and layer suffix
        base_name = base_model
        if "_attention_last" in base_model:
            base_name = base_model.replace("_attention_last", "")
        elif "_attention_all" in base_model:
            base_name = base_model.replace("_attention_all", "")
        elif "_linear_last" in base_model:
            base_name = base_model.replace("_linear_last", "")
        elif "_linear_all" in base_model:
            base_name = base_model.replace("_linear_all", "")

        for probe_type, layers in expected_combinations:
            # Create new base model name for this combination
            # Remove "weighted_" prefix for the base model name
            clean_probe_type = probe_type.replace("weighted_", "")
            new_base_model = f"{base_name}_{clean_probe_type}_{layers}"

            for dataset in datasets:
                # Check if this combination exists
                existing = df[
                    (df["base_model"] == base_model)
                    & (df["dataset_name"] == dataset)
                    & (df["probe_type"] == probe_type)
                    & (df["layers"] == layers)
                ]

                if len(existing) > 0:
                    # Use existing data but update base_model name
                    row = existing.iloc[0].copy()
                    row["base_model"] = new_base_model
                else:
                    # Create placeholder row
                    row = pd.Series(
                        {
                            "base_model": new_base_model,
                            "dataset_name": dataset,
                            "probe_type": probe_type,
                            "layers": layers,
                            "metric": np.nan,
                            "benchmark": (
                                df["benchmark"].iloc[0] if "benchmark" in df.columns else "unknown"
                            ),
                        }
                    )
                all_rows.append(row)

    return pd.DataFrame(all_rows)


def _pivot_long_to_wide_with_probe_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-form metrics to wide form with datasets as columns, including probe info.

    First ensures all 4 combinations exist for each base model, then pivots.
    Adds probe_type and layers as columns after base_model.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form dataframe filtered to a single benchmark.

    Returns
    -------
    pd.DataFrame
        Wide-form dataframe with index `base_model` and columns as datasets,
        plus probe_type and layers columns.
    """
    # Ensure all combinations exist
    df_complete = _create_base_model_combinations(df)

    # Pivot to wide format
    grouped = (
        df_complete[["base_model", "dataset_name", "metric"]]
        .groupby(["base_model", "dataset_name"], as_index=False)
        .agg({"metric": "mean"})
    )
    wide = grouped.pivot(index="base_model", columns="dataset_name", values="metric")
    wide = wide.sort_index(axis=0).sort_index(axis=1)

    # Add probe_type and layers columns
    probe_info = df_complete[["base_model", "probe_type", "layers"]].drop_duplicates()
    probe_info = probe_info.set_index("base_model")
    wide = wide.join(probe_info, how="left")

    # Add ssl column (binary) computed from base_model index
    ssl_flags = wide.index.to_series().apply(_compute_ssl_flag)
    wide["ssl"] = ssl_flags

    # Reorder columns to put probe_type and layers after base_model
    cols = list(wide.columns)
    if "probe_type" in cols and "layers" in cols and "ssl" in cols:
        # Remove from current position
        cols = [c for c in cols if c not in ["probe_type", "layers", "ssl"]]
        # Insert after base_model (which is the index)
        probe_cols = ["probe_type", "layers", "ssl"]
        wide = wide[probe_cols + cols]

    # Ensure float dtype for metric columns and round to 2 decimal places
    metric_cols = [c for c in wide.columns if c not in ["probe_type", "layers", "ssl"]]
    wide[metric_cols] = wide[metric_cols].astype(float).round(2)

    return wide


def _fit_linear(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Fit a simple linear regression y = a * x + b using least squares.

    Parameters
    ----------
    x : np.ndarray
        Predictor values (1D).
    y : np.ndarray
        Target values (1D).

    Returns
    -------
    tuple of float or None
        (a, b) coefficients if fit succeeds and has >= 3 points, else None.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.shape[0] < 3:
        return None
    try:
        a, b = np.polyfit(x_fit, y_fit, deg=1)
        return float(a), float(b)
    except Exception:
        return None


def _predict_from_pairwise_models(
    wide: pd.DataFrame,
    row_idx: int,
    target_col: str,
    min_points: int = 3,
) -> Optional[float]:
    """
    Predict a missing cell using pairwise linear regressions between datasets.

    For the target dataset column, we fit y=target, x=source for each source
    dataset that has a value in the current row. Each model is trained using
    all rows where both columns are present. Predictions from all valid models
    are averaged to produce the final estimate.

    Parameters
    ----------
    wide : pd.DataFrame
        Wide-form dataframe (rows: base_model, columns: datasets).
    row_idx : int
        Row integer index for which to predict the missing value.
    target_col : str
        Target dataset column name to predict.
    min_points : int, optional
        Minimum number of paired points required to fit a model.

    Returns
    -------
    float or None
        Predicted value if at least one model is valid, else None.
    """
    predictions: List[float] = []
    # Value of other datasets for this row
    row_series = wide.iloc[row_idx]
    for src_col, x_i in row_series.items():
        if src_col == target_col or not np.isfinite(x_i):
            continue
        # Training data across rows for this pair (src -> target)
        x_all = wide[src_col].to_numpy(dtype=float)
        y_all = wide[target_col].to_numpy(dtype=float)
        # Fit model
        coeffs = _fit_linear(x_all, y_all)
        if coeffs is None:
            continue
        a, b = coeffs
        y_hat = a * float(x_i) + b
        if np.isfinite(y_hat):
            predictions.append(float(y_hat))
    if not predictions:
        return None
    return float(np.mean(predictions))


def interpolate_missing(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing cells using a regression-based ensemble:
    1) Within-base-model Ridge over probe_type/layers
    2) Global RandomForest over hand-crafted features
    3) Baseline mean by (probe_type, layers)

    Parameters
    ----------
    wide : pd.DataFrame
        Wide-form dataframe with NaNs for missing cells.

    Returns
    -------
    pd.DataFrame
        Dataframe with NaNs filled using regression ensemble.
    """
    filled = wide.copy()

    # Only interpolate metric columns (exclude probe_type and layers)
    metric_cols = [c for c in filled.columns if c not in ["probe_type", "layers", "ssl"]]
    metric_df = filled[metric_cols]

    # Quick exit
    if metric_df.isna().sum().sum() == 0:
        return filled

    # Apply similarity prefill to help completely empty rows first
    filled = _interpolate_by_similarity(filled, metric_cols)

    # Build helper structures once
    base_models = list(filled.index)
    probe_type_array = (filled["probe_type"] == "weighted_attention").astype(int).to_numpy()
    layers_array = (filled["layers"] == "all").astype(int).to_numpy()
    ssl_array = filled["ssl"].astype(int).to_numpy()

    # Global feature builder
    def _extract_base_features(name: str) -> np.ndarray:
        name_low = name.lower()
        arch = [
            1 if "beats" in name_low else 0,
            1 if "eat" in name_low else 0,
            1 if "bird" in name_low else 0,
            1
            if ("beats" not in name_low and "eat" not in name_low and "bird" not in name_low)
            else 0,
        ]
        train = [
            1 if "pretrained" in name_low else 0,
            1 if "finetuned" in name_low else 0,
            1 if "ssl" in name_low else 0,
        ]
        return np.array(arch + train, dtype=float)

    base_features_matrix = np.vstack([_extract_base_features(b) for b in base_models])
    global_features = np.concatenate(
        [
            base_features_matrix,
            probe_type_array[:, None],
            layers_array[:, None],
            ssl_array[:, None],
        ],
        axis=1,
    )

    # For each dataset column, fit predictors using available rows and fill NaNs
    for dataset in metric_cols:
        y = filled[dataset].to_numpy()

        # Indices with observed values
        observed_idx = np.where(np.isfinite(y))[0]
        missing_idx = np.where(~np.isfinite(y))[0]
        if observed_idx.size == 0 or missing_idx.size == 0:
            continue

        # Baseline means by (probe_type, layers)
        baseline_values: Dict[Tuple[int, int], float] = {}
        for pt in [0, 1]:
            for lay in [0, 1]:
                mask = (probe_type_array == pt) & (layers_array == lay) & np.isfinite(y)
                if np.any(mask):
                    baseline_values[(pt, lay)] = float(np.nanmean(y[mask]))

        # Global model
        X_obs_global = global_features[observed_idx]
        y_obs = y[observed_idx]
        global_model = RandomForestRegressor(n_estimators=200, random_state=42)
        global_model.fit(X_obs_global, y_obs)

        # Within-base-model model (Ridge over probe/layers)
        # We train one ridge per base family if it has >=2 observed configs
        predictions: Dict[int, List[float]] = {}

        # Predict for missing with global model
        X_miss_global = global_features[missing_idx]
        global_pred = global_model.predict(X_miss_global)
        for i, idx in enumerate(missing_idx):
            predictions.setdefault(idx, []).append(float(global_pred[i]))

        # Group rows by base family and run ridge if possible
        base_families: Dict[str, List[int]] = {}
        for i, bm in enumerate(base_models):
            fam = _extract_base_family(bm)
            base_families.setdefault(fam, []).append(i)

        for _fam, rows in base_families.items():
            rows_arr = np.array(rows)
            fam_obs = rows_arr[np.isfinite(y[rows_arr])]
            if fam_obs.size < 2:
                continue
            X_fam = np.column_stack(
                [
                    probe_type_array[rows_arr],
                    layers_array[rows_arr],
                    ssl_array[rows_arr],
                ]
            )
            y_fam = y[rows_arr]
            mask_obs = np.isfinite(y_fam)
            if mask_obs.sum() < 2:
                continue
            model = Ridge(alpha=1.0)
            model.fit(X_fam[mask_obs], y_fam[mask_obs])

            # Predict for missing rows in this family
            fam_miss_mask = ~np.isfinite(y_fam)
            if fam_miss_mask.sum() > 0:
                fam_preds = model.predict(X_fam[fam_miss_mask])
                miss_global_idxs = rows_arr[fam_miss_mask]
                for i, idx in enumerate(miss_global_idxs):
                    predictions.setdefault(idx, []).append(float(fam_preds[i]))

        # Baseline fallback by probe/layers
        for idx in missing_idx:
            pt = probe_type_array[idx]
            lay = layers_array[idx]
            if (pt, lay) in baseline_values:
                predictions.setdefault(idx, []).append(baseline_values[(pt, lay)])

        # Write back averaged predictions, clipped to [0,1]
        for idx, preds in predictions.items():
            if not preds:
                continue
            val = float(np.mean(preds))
            val = max(0.0, min(1.0, val))
            filled.iat[idx, filled.columns.get_loc(dataset)] = round(val, 2)

    # Round metrics at the end as well
    filled[metric_cols] = filled[metric_cols].round(2)

    return filled


def _interpolate_by_similarity(wide: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Interpolate missing values by finding similar models and probe types.

    Parameters
    ----------
    wide : pd.DataFrame
        Wide-form dataframe with NaNs for missing cells.
    metric_cols : List[str]
        List of metric column names.

    Returns
    -------
    pd.DataFrame
        Dataframe with similarity-based interpolations applied.
    """
    filled = wide.copy()

    for idx, row in filled.iterrows():
        # Check if this row has any missing values
        if not row[metric_cols].isna().any():
            continue

        base_model = idx  # base_model is the index
        probe_type = row["probe_type"]
        layers = row["layers"]

        # Find similar models (same base model family)
        base_family = _extract_base_family(base_model)
        similar_models = filled[
            (filled.index.str.contains(base_family, case=False, na=False))
            | (filled["probe_type"] == probe_type)
            | (filled["layers"] == layers)
        ]

        # If we have similar models with data, use their average
        if len(similar_models) > 0:
            similar_data = similar_models[metric_cols]
            # Only use rows that have some data
            similar_data = similar_data.dropna(how="all")

            if len(similar_data) > 0:
                # Calculate mean of similar models
                mean_values = similar_data.mean()

                # Fill missing values with the mean
                for col in metric_cols:
                    if pd.isna(row[col]) and not pd.isna(mean_values[col]):
                        filled.at[idx, col] = mean_values[col]

    return filled


def _extract_base_family(base_model: str) -> str:
    """
    Extract the base family name from a base model.

    Parameters
    ----------
    base_model : str
        Base model name like "eat_hf_finetuned_attention_all"

    Returns
    -------
    str
        Base family name like "eat_hf_finetuned"
    """
    # Remove probe type and layer suffixes
    model = base_model
    for suffix in [
        "_attention_all",
        "_attention_last_layer",
        "_linear_all",
        "_linear_last_layer",
    ]:
        if model.endswith(suffix):
            model = model[: -len(suffix)]
            break

    return model


def _compute_ssl_flag(base_model: str) -> int:
    """
    Compute SSL binary flag from base_model name using provided rules.

    Rules
    -----
    - Names containing 'ssl_' => 1
    - Names containing 'sl_' => 0
    - Names containing 'efficientnet' => 0
    - Names containing 'bird_aves' => 1
    - Names containing 'beats_pretrained' => 1
    - Names containing 'beats_naturelm' => 0

    Parameters
    ----------
    base_model : str
        Base model name

    Returns
    -------
    int
        1 if SSL, else 0
    """
    name = base_model.lower()
    # Precedence: names starting with sl_ enforce 0; otherwise ssl_ enforces 1
    if name.startswith("sl_"):
        return 0
    if "ssl_" in name:
        return 1
    # Any model containing _pretrained should be SSL=1
    if "_pretrained" in name:
        return 1
    if "efficientnet" in name:
        return 0
    if "bird_aves" in name:
        return 1
    if "beats_naturelm" in name:
        return 0
    # Default to 0 if unknown
    return 0


def build_output_path(input_path: str, interpolate: bool) -> str:
    """
    Build a default output path based on the input file name and flags.

    Parameters
    ----------
    input_path : str
        Input CSV path.
    interpolate : bool
        Whether interpolation is enabled.

    Returns
    -------
    str
        Output CSV path.
    """
    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    suffix = "_wide_interpolated.csv" if interpolate else "_wide.csv"
    return os.path.join(os.path.dirname(input_path), name + suffix)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pivot extracted metrics CSV (long) into wide format with "
            "datasets as columns. "
            "Optionally interpolate missing cells using pairwise linear regressions."
        )
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help=(
            "Path to extracted metrics CSV (e.g., evaluation_results/extracted_metrics_beans.csv)"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional output CSV path (default derives from input file)",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Enable interpolation to fill missing dataset values",
    )
    return parser.parse_args()


def _split_beans_by_task(wide: pd.DataFrame, output_dir: str) -> None:
    """
    Split beans wide table into classification and detection datasets.

    Parameters
    ----------
    wide : pd.DataFrame
        Wide-form dataframe with beans data.
    output_dir : str
        Directory to save the split CSV files.
    """
    # Identify classification and detection columns
    classification_cols = [col for col in wide.columns if col.endswith("_classification")]
    detection_cols = [col for col in wide.columns if col.endswith("_detection")]

    # Metadata columns (probe_type, layers, ssl)
    metadata_cols = ["probe_type", "layers", "ssl"]

    print(f"Found {len(classification_cols)} classification datasets: {classification_cols}")
    print(f"Found {len(detection_cols)} detection datasets: {detection_cols}")

    # Create classification table
    if classification_cols:
        classification_df = wide[metadata_cols + classification_cols].copy()
        classification_output = os.path.join(
            output_dir, "extracted_metrics_beans_classification.csv"
        )
        classification_df.to_csv(classification_output)
        print(
            f"Saved classification table with shape {classification_df.shape} "
            f"to {classification_output}"
        )

    # Create detection table
    if detection_cols:
        detection_df = wide[metadata_cols + detection_cols].copy()
        detection_output = os.path.join(output_dir, "extracted_metrics_beans_detection.csv")
        detection_df.to_csv(detection_output)
        print(f"Saved detection table with shape {detection_df.shape} to {detection_output}")


def main() -> None:
    """
    Entry point to pivot long-form metrics into wide format and optionally interpolate.

    Steps
    -----
    1. Read long-form CSV and validate columns.
    2. Detect and enforce single benchmark.
    3. Pivot to wide with datasets as columns and base models as rows.
    4. Optionally interpolate missing cells via pairwise regressions between datasets.
    5. For beans benchmark: split into classification and detection CSV files.
    6. Save to CSV.
    """
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    _validate_input_columns(df)
    bench = _detect_benchmark(df)
    df_bench = df[df["benchmark"] == bench].copy()
    wide = _pivot_long_to_wide_with_probe_info(df_bench)
    if args.interpolate:
        wide = interpolate_missing(wide)

    # Save main wide table
    out_path = args.output or build_output_path(args.input_csv, args.interpolate)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wide.to_csv(out_path)
    print(f"Wrote wide table with shape {wide.shape} to {out_path}")

    # For beans benchmark, also create task-specific splits
    if bench == "beans":
        output_dir = os.path.dirname(out_path) or "."
        _split_beans_by_task(wide, output_dir)


if __name__ == "__main__":
    main()
