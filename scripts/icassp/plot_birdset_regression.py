"""Plot BirdSet regression-wide metrics and summarize best configs.

This script reads a CSV of probing results in a wide format (with the first
four columns being metadata: ``base_model``, ``probe_type``, ``layers``,
``ssl``) followed by dataset columns. It produces a grouped bar chart that
compares average performance across all datasets for each model and probe
configuration, and prints a concise summary of the best configuration per
base model.

Example
-------
Run with uv to ensure the project environment is used.

```bash
uv run python scripts/plot_birdset_regression.py \
  --csv evaluation_results/extracted_metrics_birdset_wide_regression.csv \
  --out plots/birdset_regression.png
```
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _dataset_columns(columns: Iterable[str]) -> list[str]:
    """Return dataset metric column names.

    Parameters
    ----------
    columns
        Iterable of CSV column names.

    Returns
    -------
    list[str]
        All columns after the first four metadata fields.

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    cols = list(columns)
    if len(cols) < 5:
        raise ValueError(
            "Expected at least 5 columns: base_model, probe_type, layers, ssl, <metric>",
        )

    # Exclude non-metric columns such as the auxiliary fully_ft flag.
    meta = {"base_model", "probe_type", "layers", "ssl", "fully_ft"}
    metric_cols = [c for c in cols if c not in meta]
    if not metric_cols:
        raise ValueError("No metric columns found after excluding metadata.")
    return metric_cols


def _clean_base_model_name(base_model: str) -> str:
    """Remove probe and layer suffixes from a base model string.

    Parameters
    ----------
    base_model
        Base model string possibly containing probe/layer suffixes.

    Returns
    -------
    str
        Cleaned base model name.
    """

    return (
        base_model.replace("_attention_all", "")
        .replace("_attention_last_layer", "")
        .replace("_linear_all", "")
        .replace("_linear_last_layer", "")
        .replace("_attention_ft", "")
        .replace("_linear_ft", "")
        .removesuffix("_ft")
    )


def create_probing_comparison_plot(
    csv_file_path: str | Path,
    output_path: str | Path | None = None,
    figsize: Tuple[float, float] = (14.0, 8.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a compact probing comparison plot from CSV data.

    Parameters
    ----------
    csv_file_path
        Path to the CSV file.
    output_path
        Optional path to save the plot image.
    figsize
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created figure and axes.
    """

    df = pd.read_csv(csv_file_path)
    _dataset_columns(df.columns)

    def process_data(frame: pd.DataFrame, ds_cols: list[str]) -> pd.DataFrame:
        results: list[dict[str, object]] = []
        for _, row in frame.iterrows():
            base_model_raw: str = row["base_model"]
            base_model = _clean_base_model_name(base_model_raw)

            # Exclude specified base models entirely from the heatmap
            if base_model in {"eat_hf_pretrained", "eat_hf_finetuned"}:
                continue

            # Temporary exclusion of specific base models from the heatmap
            if base_model in {"eat_hf_pretrained", "eat_hf_finetuned"}:
                continue

            probe_type = "Attention" if "attention" in str(row["probe_type"]).lower() else "Linear"
            layers = "All" if str(row["layers"]) == "all" else "Last"
            ssl = "SSL" if int(row["ssl"]) == 1 else "SL"

            dataset_scores = [float(row[col]) for col in ds_cols]
            avg_score = float(np.mean(dataset_scores))

            results.append(
                {
                    "base_model": base_model,
                    "ssl": ssl,
                    "probe_type": probe_type,
                    "layers": layers,
                    "avg_score": avg_score,
                    "model_label": f"{base_model.replace('_', ' ')}\n({ssl})",
                }
            )

        return pd.DataFrame(results)

    processed_df = process_data(df)

    pivot_df = processed_df.pivot_table(
        index="model_label",
        columns=["probe_type", "layers"],
        values="avg_score",
        fill_value=0.0,
    )
    pivot_df = pivot_df.sort_index()

    # Ensure each display model appears in both SSL and SL groups by
    # duplicating the existing row if the counterpart is missing.
    # Extract short display names (before the newline and supervision tag)
    index_names = list(pivot_df.index)
    short_to_tags: dict[str, set[str]] = {}
    for lbl in index_names:
        short = str(lbl).split("\n")[0]
        tag = "SSL" if "(SSL)" in str(lbl) else "SL"
        short_to_tags.setdefault(short, set()).add(tag)

    rows_to_add: list[tuple[str, pd.Series]] = []
    for lbl in index_names:
        short = str(lbl).split("\n")[0]
        present = short_to_tags.get(short, set())
        if "SSL" in present and "SL" not in present:
            new_idx = f"{short}\n(SL)"
            rows_to_add.append((new_idx, pivot_df.loc[lbl]))
        elif "SL" in present and "SSL" not in present:
            new_idx = f"{short}\n(SSL)"
            rows_to_add.append((new_idx, pivot_df.loc[lbl]))

    if rows_to_add:
        for new_idx, row_values in rows_to_add:
            pivot_df.loc[new_idx] = row_values
        pivot_df = pivot_df.sort_index()

    # Group rows by supervision type (SSL first, then SL) and compute counts
    row_labels_hm = list(pivot_df.index)
    ssl_idx_hm = [i for i, lbl in enumerate(row_labels_hm) if "(SSL)" in lbl]
    sl_idx_hm = [i for i, lbl in enumerate(row_labels_hm) if "(SL)" in lbl]
    new_order_hm = ssl_idx_hm + sl_idx_hm
    if new_order_hm and len(new_order_hm) == len(row_labels_hm):
        pivot_df = pivot_df.iloc[new_order_hm]
    ssl_count_hm = sum(1 for lbl in pivot_df.index if "(SSL)" in lbl)
    _sl_count_hm = len(pivot_df.index) - ssl_count_hm

    # Group rows by supervision type (SL first, then SSL)
    row_labels = list(pivot_df.index)
    sl_indices_local = [i for i, lbl in enumerate(row_labels) if "(SL)" in lbl]
    ssl_indices_local = [i for i, lbl in enumerate(row_labels) if "(SSL)" in lbl]
    new_order = sl_indices_local + ssl_indices_local
    if new_order and len(new_order) == len(row_labels):
        pivot_df = pivot_df.iloc[new_order]
    # Counts after potential reordering
    sl_count = sum(1 for lbl in pivot_df.index if "(SL)" in lbl)
    _ssl_count = len(pivot_df.index) - sl_count

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        ("Attention", "All"): "#1e40af",
        ("Attention", "Last"): "#60a5fa",
        ("Linear", "All"): "#dc2626",
        ("Linear", "Last"): "#f87171",
    }

    x_pos = np.arange(len(pivot_df.index), dtype=float)
    bar_width = 0.2

    configs = [
        ("Attention", "All"),
        ("Attention", "Last"),
        ("Linear", "All"),
        ("Linear", "Last"),
    ]
    config_names = [
        "Attention (All Layers)",
        "Attention (Last Layer)",
        "Linear (All Layers)",
        "Linear (Last Layer)",
    ]

    for i, (config, name) in enumerate(zip(configs, config_names, strict=True)):
        if config in pivot_df.columns:
            values = pivot_df[config].values
            bar_positions = x_pos + (i - 1.5) * bar_width
            ax.bar(
                bar_positions,
                values,
                bar_width,
                label=name,
                color=colors[config],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.6,
            )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Performance", fontsize=12, fontweight="bold")
    ax.set_title(
        "Average Performance Across BirdSet Datasets by Model and Probe",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right", fontsize=9)

    max_val = float(pivot_df.values.max()) if pivot_df.size else 1.0
    ax.set_ylim(0.0, max_val * 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{x:.2f}"))

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    explanation = (
        "SSL: Self-supervised, SL: Supervised\n"
        "Performance: Average across all BirdSet detection datasets\n"
        "Colors: Blue = Attention probes, Red = Linear probes\n"
        "Shades: Dark = All layers, Light = Last layer"
    )
    ax.text(
        0.02,
        0.98,
        explanation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        LOGGER.info("Plot saved to %s", output_path)

    return fig, ax


def create_summary_table(csv_file_path: str | Path) -> pd.DataFrame:
    """Create a summary table showing best configurations per base model.

    Parameters
    ----------
    csv_file_path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Summary table with best configuration per base model.
    """

    df = pd.read_csv(csv_file_path)
    dataset_cols = _dataset_columns(df.columns)

    df = df.copy()
    df["avg_performance"] = df[dataset_cols].mean(axis=1)

    df["base_model_clean"] = df["base_model"].str.replace(r"_(attention|linear)_(all|last_layer)$", "", regex=True)

    idx = df.groupby("base_model_clean")["avg_performance"].idxmax()
    best_configs = df.loc[idx]

    summary = best_configs[["base_model_clean", "probe_type", "layers", "ssl", "avg_performance"]].copy()
    summary["ssl_label"] = summary["ssl"].map({0: "Supervised", 1: "Self-supervised"})
    summary = summary.sort_values("avg_performance", ascending=False)

    return summary


def create_probing_heatmap(
    csv_file_path: str | Path,
    output_path: str | Path | None = None,
    figsize: Tuple[float, float] = (14.0, 8.0),
    csv_beans_classification: str | Path | None = None,
    csv_beans_detection: str | Path | None = None,
    include_ft: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a heatmap with models as rows and probe configs as columns.

    The cell values represent the average performance across all dataset
    columns for the corresponding model and probe configuration.

    Parameters
    ----------
    csv_file_path
        Path to the CSV file.
    output_path
        Optional path to save the plot image.
    figsize
        Figure size ``(width, height)`` in inches.
    csv_beans_classification
        Optional path to Beans Classification CSV.
    csv_beans_detection
        Optional path to Beans Detection CSV.
    include_ft
        If True, include fully fine-tuned (FT) models, resulting in 6 columns
        per dataset (Attention/Linear × All/Last/FT). If False, exclude FT
        models for the original 4-column plot (Attention/Linear × All/Last).

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created figure and axes.
    """

    df = pd.read_csv(csv_file_path)
    dataset_cols = _dataset_columns(df.columns)

    def process_data(frame: pd.DataFrame, ds_cols: list[str]) -> pd.DataFrame:
        results: list[dict[str, object]] = []
        name_map = {
            "beats_naturelm": "NatureBEATs",
            "eat_hf_finetuned": "EAT_excluded",
            "eat_hf_pretrained": "EAT_excluded",
            "efficientnet_animalspeak_audioset": "EfficientNet",
            "sl_beats_all": "BEATs",
            "beats_pretrained": "BEATs",
            "bird_aves_bio": "AVES",
            # "sl_eat_all_ssl_all": "EATall",
            # "ssl_eat_all": "EATall",
            "sl_eat_all_ssl_all": "EAT",
            "ssl_eat_all": "EAT",
        }
        for _, row in frame.iterrows():
            base_model_raw: str = row["base_model"]
            base_model = _clean_base_model_name(base_model_raw)

            probe_type = "Attention" if "attention" in str(row["probe_type"]).lower() else "Linear"
            # Check if this is a fully fine-tuned model
            is_ft = False
            if "fully_ft" in row and pd.notna(row["fully_ft"]) and bool(row["fully_ft"]):
                is_ft = True
            elif base_model_raw.endswith("_ft"):
                is_ft = True

            # Skip FT models if include_ft is False
            if is_ft and not include_ft:
                continue

            if is_ft:
                layers = "FT"
            elif str(row["layers"]) == "all":
                layers = "All"
            else:
                layers = "Last"

            # Default SSL/SL from CSV, then override for specific models
            ssl_tag = "SSL" if int(row["ssl"]) == 1 else "SL"
            ssl_overrides = {
                "eat_hf_finetuned": "SL",
                "eat_hf_pretrained": "SSL",
                "ssl_eat_all": "SSL",
                "sl_eat_all_ssl_all": "SL",
                "beats_naturelm": "SL",
            }
            if base_model in ssl_overrides:
                ssl_tag = ssl_overrides[base_model]

            dataset_scores = [float(row[col]) for col in ds_cols]
            avg_score = float(np.mean(dataset_scores))

            display = name_map.get(base_model, base_model.replace("_", " "))
            model_label = f"{display}\n({ssl_tag})"

            results.append(
                {
                    "base_model": base_model,
                    "ssl_tag": ssl_tag,
                    "probe_type": probe_type,
                    "layers": layers,
                    "avg_score": avg_score,
                    "model_label": model_label,
                }
            )

        return pd.DataFrame(results)

    # Build per-dataset pivots (BirdSet is always provided)
    def build_pivot(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        proc = process_data(frame, cols)
        return proc.pivot_table(
            index="model_label",
            columns=["probe_type", "layers"],
            values="avg_score",
            fill_value=0.0,
        )

    # Order datasets: Beans Classification, Beans Detection, then BirdSet
    pivots: list[tuple[str, pd.DataFrame]] = []
    if csv_beans_classification:
        df_cls = pd.read_csv(csv_beans_classification)
        cols_cls = _dataset_columns(df_cls.columns)
        pivots.append(("BEANS Classification", build_pivot(df_cls, cols_cls)))
    if csv_beans_detection:
        df_det = pd.read_csv(csv_beans_detection)
        cols_det = _dataset_columns(df_det.columns)
        pivots.append(("BEANS Detection", build_pivot(df_det, cols_det)))
    pivots.append(("BirdSet", build_pivot(df, dataset_cols)))

    # Align indices across datasets (outer join over all model labels)
    all_index = pivots[0][1].index
    for _, pv in pivots[1:]:
        all_index = all_index.union(pv.index)
    aligned: list[tuple[str, pd.DataFrame]] = []
    for name, pv in pivots:
        aligned.append((name, pv.reindex(all_index)))

    # Order columns within each dataset
    if include_ft:
        # 6 combinations: 2 probe types × 3 layer options
        within_order = [
            ("Attention", "All"),
            ("Attention", "Last"),
            ("Attention", "FT"),
            ("Linear", "All"),
            ("Linear", "Last"),
            ("Linear", "FT"),
        ]
        cols_per_dataset = 6
    else:
        # 4 combinations: 2 probe types × 2 layer options (original)
        within_order = [
            ("Attention", "All"),
            ("Attention", "Last"),
            ("Linear", "All"),
            ("Linear", "Last"),
        ]
        cols_per_dataset = 4

    # Concatenate with a first-level for dataset
    parts: list[pd.DataFrame] = []
    for name, pv in aligned:
        cols_in_ds = [c for c in within_order if c in pv.columns]
        pv_ordered = pv[cols_in_ds]
        new_cols = [(name, c[0], c[1]) for c in pv_ordered.columns]
        pv_ordered.columns = pd.MultiIndex.from_tuples(new_cols, names=["dataset", "probe_type", "layers"])
        parts.append(pv_ordered)
    combined = pd.concat(parts, axis=1)
    combined = combined.sort_index()

    # Temporary hard filter: drop EAT (both SSL/SL) rows if present
    combined = combined[~combined.index.to_series().str.startswith("EAT_excluded\n")]

    # Order rows: SSL first, then SL (based on label text)
    labels = list(combined.index)
    ssl_labels = [lbl for lbl in labels if "(SSL)" in str(lbl)]
    sl_labels = [lbl for lbl in labels if "(SL)" in str(lbl)]
    combined = combined.loc[ssl_labels + sl_labels]
    ssl_count_hm = len(ssl_labels)
    sl_count_hm = len(sl_labels)

    data = combined.values
    # Normalize colors per dataset block
    num_datasets = len(parts)
    norm_data = data.copy().astype(float)
    for k in range(num_datasets):
        j0, j1 = k * cols_per_dataset, k * cols_per_dataset + cols_per_dataset
        block = norm_data[:, j0:j1]
        block_min = float(np.min(block))
        block_max = float(np.max(block))
        if block_max > block_min:
            norm_data[:, j0:j1] = (block - block_min) / (block_max - block_min)
        else:
            norm_data[:, j0:j1] = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    # Render colored cells and shrink right margin for group labels
    n_rows, n_cols = data.shape
    im = ax.imshow(norm_data, aspect="auto", cmap="viridis")
    ax.set_xlim(-0.5, n_cols - 0.5)

    # Labels (already include display name + supervision tag)
    ax.set_yticks(np.arange(combined.shape[0]))
    ax.set_yticklabels(combined.index, fontsize=18, fontweight="bold")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right")

    # Build repeated col labels per dataset
    xticks = np.arange(data.shape[1])
    labels_x: list[str] = []
    if include_ft:
        label_set = [
            "Attention\n(All)",
            "Attention\n(Last)",
            "Attention\n(FT)",
            "Linear\n(All)",
            "Linear\n(Last)",
            "Linear\n(FT)",
        ]
    else:
        label_set = [
            "Attention\n(All)",
            "Attention\n(Last)",
            "Linear\n(All)",
            "Linear\n(Last)",
        ]
    for _k in range(num_datasets):
        labels_x.extend(label_set)
    x_tick_fontsize = 14 if not include_ft else 10
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels_x, fontsize=x_tick_fontsize, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add dataset group labels centered over each dataset block
    dataset_names = [name for name, _ in aligned]
    for k, ds_name in enumerate(dataset_names):
        x_center = k * cols_per_dataset + (cols_per_dataset - 1) / 2  # center of block in data coords
        x_frac = (x_center + 0.5) / n_cols  # convert to axes fraction
        ax.text(
            x_frac,
            1.02,
            ds_name,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
        )

    # Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.grid(False)
    # Remove explicit grid lines (heatmap cells provide structure)

    # Draw a single red separator line between SSL (top) and SL (bottom) groups
    if sl_count_hm > 0 and ssl_count_hm > 0:
        boundary = ssl_count_hm
        x_start, x_end = -0.5, n_cols - 0.5
        y_mid = boundary - 0.5
        ax.hlines(y_mid, x_start, x_end, colors="red", linewidth=2.0)
    # No right-side group labels; supervision shown in y-ticks
    # Draw vertical single red separators between datasets
    if num_datasets > 1:
        for k in range(1, num_datasets):
            x_boundary = (k * cols_per_dataset) - 0.5
            y_start, y_end = -0.5, n_rows - 0.5
            ax.vlines(x_boundary, y_start, y_end, colors="red", linewidth=2.0)
    # Annotate each cell with its value
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = float(data[i, j])
            # Choose text color based on normalized value for contrast
            color = "white" if im.norm(val) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=18,
            )
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        LOGGER.info("Plot saved to %s", output_path)

    return fig, ax


def create_layer_wise_heatmap_ssl(
    csv_file_path: str | Path,
    output_path: str | Path | None = None,
    figsize: Tuple[float, float] = (14.0, 8.0),
    csv_beans_classification: str | Path | None = None,
    csv_beans_detection: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a layer-wise heatmap for SSL methods only.

    Parameters
    ----------
    csv_file_path
        Path to the CSV file.
    output_path
        Optional path to save the plot image.
    figsize
        Figure size ``(width, height)`` in inches.
    csv_beans_classification
        Optional path to Beans Classification CSV.
    csv_beans_detection
        Optional path to Beans Detection CSV.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created figure and axes.
    """

    df = pd.read_csv(csv_file_path)
    dataset_cols = _dataset_columns(df.columns)

    def process_data(frame: pd.DataFrame, ds_cols: list[str]) -> pd.DataFrame:
        results: list[dict[str, object]] = []
        name_map = {
            "beats_naturelm": "NatureBEATs",
            "eat_hf_finetuned": "EAT_excluded",
            "eat_hf_pretrained": "EAT_excluded",
            "efficientnet_animalspeak_audioset": "EfficientNet",
            "sl_beats_all": "BEATs",
            "beats_pretrained": "BEATs",
            "bird_aves_bio": "AVES",
            "sl_eat_all_ssl_all": "EAT",
            "ssl_eat_all": "EAT",
        }
        for _, row in frame.iterrows():
            base_model_raw: str = row["base_model"]
            base_model = _clean_base_model_name(base_model_raw)

            # Exclude specified base models
            if base_model in {"eat_hf_pretrained", "eat_hf_finetuned"}:
                continue

            probe_type = "Attention" if "attention" in str(row["probe_type"]).lower() else "Linear"
            # Check if this is a fully fine-tuned model
            is_ft = False
            if "fully_ft" in row and pd.notna(row["fully_ft"]) and bool(row["fully_ft"]):
                is_ft = True
            elif base_model_raw.endswith("_ft"):
                is_ft = True

            # Filter out FT models in layer-wise heatmaps
            if is_ft:
                continue

            if str(row["layers"]) == "all":
                layers = "All"
            else:
                layers = "Last"

            ssl_tag = "SSL" if int(row["ssl"]) == 1 else "SL"
            ssl_overrides = {
                "eat_hf_finetuned": "SL",
                "eat_hf_pretrained": "SSL",
                "ssl_eat_all": "SSL",
                "sl_eat_all_ssl_all": "SL",
            }
            if base_model in ssl_overrides:
                ssl_tag = ssl_overrides[base_model]

            # Only include SSL methods
            if ssl_tag != "SSL":
                continue

            dataset_scores = [float(row[col]) for col in ds_cols]
            avg_score = float(np.mean(dataset_scores))

            display = name_map.get(base_model, base_model.replace("_", " "))
            model_label = f"{display}"

            results.append(
                {
                    "base_model": base_model,
                    "ssl_tag": ssl_tag,
                    "probe_type": probe_type,
                    "layers": layers,
                    "avg_score": avg_score,
                    "model_label": model_label,
                }
            )

        return pd.DataFrame(results)

    # Build per-dataset pivots
    def build_pivot(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        proc = process_data(frame, cols)
        return proc.pivot_table(
            index="model_label",
            columns=["probe_type", "layers"],
            values="avg_score",
            fill_value=0.0,
        )

    # Order datasets: Beans Classification, Beans Detection, then BirdSet
    pivots: list[tuple[str, pd.DataFrame]] = []
    if csv_beans_classification:
        df_cls = pd.read_csv(csv_beans_classification)
        cols_cls = _dataset_columns(df_cls.columns)
        pivots.append(("BEANS Classification", build_pivot(df_cls, cols_cls)))
    if csv_beans_detection:
        df_det = pd.read_csv(csv_beans_detection)
        cols_det = _dataset_columns(df_det.columns)
        pivots.append(("BEANS Detection", build_pivot(df_det, cols_det)))
    pivots.append(("BirdSet", build_pivot(df, dataset_cols)))

    # Align indices across datasets
    all_index = pivots[0][1].index
    for _, pv in pivots[1:]:
        all_index = all_index.union(pv.index)
    aligned: list[tuple[str, pd.DataFrame]] = []
    for name, pv in pivots:
        aligned.append((name, pv.reindex(all_index)))

    # Order columns within each dataset (4 combinations: 2 probe types × 2 layer options, no FT)
    within_order = [
        ("Attention", "All"),
        ("Attention", "Last"),
        ("Linear", "All"),
        ("Linear", "Last"),
    ]

    # Concatenate with a first-level for dataset
    parts: list[pd.DataFrame] = []
    for name, pv in aligned:
        cols_in_ds = [c for c in within_order if c in pv.columns]
        pv_ordered = pv[cols_in_ds]
        new_cols = [(name, c[0], c[1]) for c in pv_ordered.columns]
        pv_ordered.columns = pd.MultiIndex.from_tuples(new_cols, names=["dataset", "probe_type", "layers"])
        parts.append(pv_ordered)
    combined = pd.concat(parts, axis=1)
    combined = combined.sort_index()

    # Filter out excluded models
    combined = combined[~combined.index.to_series().str.startswith("EAT_excluded")]

    data = combined.values
    # Normalize colors per dataset block (4 columns each, no FT)
    num_datasets = len(parts)
    norm_data = data.copy().astype(float)
    for k in range(num_datasets):
        j0, j1 = k * 4, k * 4 + 4
        block = norm_data[:, j0:j1]
        block_min = float(np.min(block))
        block_max = float(np.max(block))
        if block_max > block_min:
            norm_data[:, j0:j1] = (block - block_min) / (block_max - block_min)
        else:
            norm_data[:, j0:j1] = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    n_rows, n_cols = data.shape
    im = ax.imshow(norm_data, aspect="auto", cmap="viridis")
    ax.set_xlim(-0.5, n_cols - 0.5)

    # Labels
    ax.set_yticks(np.arange(combined.shape[0]))
    ax.set_yticklabels(combined.index, fontsize=18)

    # Build repeated col labels per dataset
    xticks = np.arange(data.shape[1])
    labels_x: list[str] = []
    for _k in range(num_datasets):
        labels_x.extend(
            [
                "Attention\n(All)",
                "Attention\n(Last)",
                "Linear\n(All)",
                "Linear\n(Last)",
            ]
        )
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels_x, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add dataset group labels centered over each 4-column block
    dataset_names = [name for name, _ in aligned]
    for k, ds_name in enumerate(dataset_names):
        x_center = k * 4 + 1.5  # center of 4-column block in data coords
        x_frac = (x_center + 0.5) / n_cols  # convert to axes fraction
        ax.text(
            x_frac,
            1.02,
            ds_name,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
        )

    ax.set_xlabel("Probe configuration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model (SSL)", fontsize=12, fontweight="bold")

    ax.grid(False)

    # Draw vertical double separators between datasets (every 4 cols)
    if num_datasets > 1:
        for k in range(1, num_datasets):
            x_boundary = (k * 4) - 0.5
            y_start, y_end = -0.5, n_rows - 0.5
            off = 0.08
            ax.vlines(x_boundary - off, y_start, y_end, colors="black", linewidth=2.0)
            ax.vlines(x_boundary + off, y_start, y_end, colors="black", linewidth=2.0)

    # Annotate each cell with its value
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = float(data[i, j])
            # Choose text color based on normalized value for contrast
            color = "white" if im.norm(val) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=18,
            )
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        LOGGER.info("Plot saved to %s", output_path)

    return fig, ax


def create_layer_wise_heatmap_sl(
    csv_file_path: str | Path,
    output_path: str | Path | None = None,
    figsize: Tuple[float, float] = (14.0, 8.0),
    csv_beans_classification: str | Path | None = None,
    csv_beans_detection: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a layer-wise heatmap for SL methods only.

    Parameters
    ----------
    csv_file_path
        Path to the CSV file.
    output_path
        Optional path to save the plot image.
    figsize
        Figure size ``(width, height)`` in inches.
    csv_beans_classification
        Optional path to Beans Classification CSV.
    csv_beans_detection
        Optional path to Beans Detection CSV.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created figure and axes.
    """

    df = pd.read_csv(csv_file_path)
    dataset_cols = _dataset_columns(df.columns)

    def process_data(frame: pd.DataFrame, ds_cols: list[str]) -> pd.DataFrame:
        results: list[dict[str, object]] = []
        name_map = {
            "beats_naturelm": "NatureBEATs",
            "eat_hf_finetuned": "EAT_excluded",
            "eat_hf_pretrained": "EAT_excluded",
            "efficientnet_animalspeak_audioset": "EfficientNet",
            "sl_beats_all": "BEATs",
            "beats_pretrained": "BEATs",
            "bird_aves_bio": "AVES",
            "sl_eat_all_ssl_all": "EAT",
            "ssl_eat_all": "EAT",
        }
        for _, row in frame.iterrows():
            base_model_raw: str = row["base_model"]
            base_model = _clean_base_model_name(base_model_raw)

            # Exclude specified base models
            if base_model in {"eat_hf_pretrained", "eat_hf_finetuned"}:
                continue

            probe_type = "Attention" if "attention" in str(row["probe_type"]).lower() else "Linear"
            # Check if this is a fully fine-tuned model
            is_ft = False
            if "fully_ft" in row and pd.notna(row["fully_ft"]) and bool(row["fully_ft"]):
                is_ft = True
            elif base_model_raw.endswith("_ft"):
                is_ft = True

            # Filter out FT models in layer-wise heatmaps
            if is_ft:
                continue

            if str(row["layers"]) == "all":
                layers = "All"
            else:
                layers = "Last"

            ssl_tag = "SSL" if int(row["ssl"]) == 1 else "SL"
            ssl_overrides = {
                "eat_hf_finetuned": "SL",
                "eat_hf_pretrained": "SSL",
                "ssl_eat_all": "SSL",
                "sl_eat_all_ssl_all": "SL",
            }
            if base_model in ssl_overrides:
                ssl_tag = ssl_overrides[base_model]

            # Only include SL methods
            if ssl_tag != "SL":
                continue

            dataset_scores = [float(row[col]) for col in ds_cols]
            avg_score = float(np.mean(dataset_scores))

            display = name_map.get(base_model, base_model.replace("_", " "))
            model_label = f"{display}"

            results.append(
                {
                    "base_model": base_model,
                    "ssl_tag": ssl_tag,
                    "probe_type": probe_type,
                    "layers": layers,
                    "avg_score": avg_score,
                    "model_label": model_label,
                }
            )

        return pd.DataFrame(results)

    # Build per-dataset pivots
    def build_pivot(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        proc = process_data(frame, cols)
        return proc.pivot_table(
            index="model_label",
            columns=["probe_type", "layers"],
            values="avg_score",
            fill_value=0.0,
        )

    # Order datasets: Beans Classification, Beans Detection, then BirdSet
    pivots: list[tuple[str, pd.DataFrame]] = []
    if csv_beans_classification:
        df_cls = pd.read_csv(csv_beans_classification)
        cols_cls = _dataset_columns(df_cls.columns)
        pivots.append(("BEANS Classification", build_pivot(df_cls, cols_cls)))
    if csv_beans_detection:
        df_det = pd.read_csv(csv_beans_detection)
        cols_det = _dataset_columns(df_det.columns)
        pivots.append(("BEANS Detection", build_pivot(df_det, cols_det)))
    pivots.append(("BirdSet", build_pivot(df, dataset_cols)))

    # Align indices across datasets
    all_index = pivots[0][1].index
    for _, pv in pivots[1:]:
        all_index = all_index.union(pv.index)
    aligned: list[tuple[str, pd.DataFrame]] = []
    for name, pv in pivots:
        aligned.append((name, pv.reindex(all_index)))

    # Order columns within each dataset (4 combinations: 2 probe types × 2 layer options, no FT)
    within_order = [
        ("Attention", "All"),
        ("Attention", "Last"),
        ("Linear", "All"),
        ("Linear", "Last"),
    ]

    # Concatenate with a first-level for dataset
    parts: list[pd.DataFrame] = []
    for name, pv in aligned:
        cols_in_ds = [c for c in within_order if c in pv.columns]
        pv_ordered = pv[cols_in_ds]
        new_cols = [(name, c[0], c[1]) for c in pv_ordered.columns]
        pv_ordered.columns = pd.MultiIndex.from_tuples(new_cols, names=["dataset", "probe_type", "layers"])
        parts.append(pv_ordered)
    combined = pd.concat(parts, axis=1)
    combined = combined.sort_index()

    # Filter out excluded models
    combined = combined[~combined.index.to_series().str.startswith("EAT_excluded")]

    data = combined.values
    # Normalize colors per dataset block (4 columns each, no FT)
    num_datasets = len(parts)
    norm_data = data.copy().astype(float)
    for k in range(num_datasets):
        j0, j1 = k * 4, k * 4 + 4
        block = norm_data[:, j0:j1]
        block_min = float(np.min(block))
        block_max = float(np.max(block))
        if block_max > block_min:
            norm_data[:, j0:j1] = (block - block_min) / (block_max - block_min)
        else:
            norm_data[:, j0:j1] = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    n_rows, n_cols = data.shape
    im = ax.imshow(norm_data, aspect="auto", cmap="viridis")
    ax.set_xlim(-0.5, n_cols - 0.5)

    # Labels
    ax.set_yticks(np.arange(combined.shape[0]))
    ax.set_yticklabels(combined.index, fontsize=18)

    # Build repeated col labels per dataset
    xticks = np.arange(data.shape[1])
    labels_x: list[str] = []
    for _k in range(num_datasets):
        labels_x.extend(
            [
                "Attention\n(All)",
                "Attention\n(Last)",
                "Linear\n(All)",
                "Linear\n(Last)",
            ]
        )
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels_x, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add dataset group labels centered over each 4-column block
    dataset_names = [name for name, _ in aligned]
    for k, ds_name in enumerate(dataset_names):
        x_center = k * 4 + 1.5  # center of 4-column block in data coords
        x_frac = (x_center + 0.5) / n_cols  # convert to axes fraction
        ax.text(
            x_frac,
            1.02,
            ds_name,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
        )

    ax.set_xlabel("Probe configuration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model (SL)", fontsize=12, fontweight="bold")

    ax.grid(False)

    # Draw vertical double separators between datasets (every 4 cols)
    if num_datasets > 1:
        for k in range(1, num_datasets):
            x_boundary = (k * 4) - 0.5
            y_start, y_end = -0.5, n_rows - 0.5
            off = 0.08
            ax.vlines(x_boundary - off, y_start, y_end, colors="black", linewidth=2.0)
            ax.vlines(x_boundary + off, y_start, y_end, colors="black", linewidth=2.0)

    # Annotate each cell with its value
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = float(data[i, j])
            # Choose text color based on normalized value for contrast
            color = "white" if im.norm(val) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=18,
            )
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        LOGGER.info("Plot saved to %s", output_path)

    return fig, ax


def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot BirdSet regression-wide probing metrics and summarize the best configuration per base model."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the BirdSet CSV file.",
    )
    parser.add_argument(
        "--csv-beans-classification",
        type=str,
        default=None,
        help="Optional path to Beans Classification CSV.",
    )
    parser.add_argument(
        "--csv-beans-detection",
        type=str,
        default=None,
        help="Optional path to Beans Detection CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save the output plot image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a window with the plot.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="bars",
        choices=["bars", "heatmap", "layer_wise_ssl", "layer_wise_sl"],
        help="Select plot type: grouped bars, heatmap, layer-wise SSL, or layer-wise SL.",
    )
    parser.add_argument(
        "--include-ft",
        action="store_true",
        default=True,
        help="Include fully fine-tuned (FT) models in the heatmap (default: True).",
    )
    parser.add_argument(
        "--no-include-ft",
        dest="include_ft",
        action="store_false",
        help="Exclude fully fine-tuned (FT) models to generate the original 4-column plot.",
    )
    return parser


def main() -> None:
    """Entry point for the plotting script."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_argparser().parse_args()

    if args.plot == "bars":
        fig, _ax = create_probing_comparison_plot(args.csv, args.out)
    elif args.plot == "heatmap":
        fig, _ax = create_probing_heatmap(
            args.csv,
            args.out,
            csv_beans_classification=args.csv_beans_classification,
            csv_beans_detection=args.csv_beans_detection,
            include_ft=args.include_ft,
        )
    elif args.plot == "layer_wise_ssl":
        fig, _ax = create_layer_wise_heatmap_ssl(
            args.csv,
            args.out,
            csv_beans_classification=args.csv_beans_classification,
            csv_beans_detection=args.csv_beans_detection,
        )
    elif args.plot == "layer_wise_sl":
        fig, _ax = create_layer_wise_heatmap_sl(
            args.csv,
            args.out,
            csv_beans_classification=args.csv_beans_classification,
            csv_beans_detection=args.csv_beans_detection,
        )

    summary = create_summary_table(args.csv)
    # Hide EAT rows in summary to match temporary heatmap filtering
    summary = summary[~summary["base_model_clean"].isin(["eat_hf_pretrained", "eat_hf_finetuned"])]
    LOGGER.info("Best Configuration per Model")
    LOGGER.info("%s", "=" * 60)
    for _, row in summary.iterrows():
        LOGGER.info(
            "%s | %s | %s | %s | %.3f",
            f"{row['base_model_clean']:<25}",
            f"{row['probe_type']:<15}",
            f"{row['layers']:<10}",
            f"{row['ssl_label']:<15}",
            float(row["avg_performance"]),
        )

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
