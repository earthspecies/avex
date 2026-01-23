#!/usr/bin/env python3
"""
Plot layer weights for each base model with error intervals across datasets.

This script extracts layer weights from the extracted metrics CSV files and creates
line plots showing the weight distribution across layers for each base model,
with error intervals computed across datasets.
"""

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_base_model_name(full_name: str) -> str:
    """Extract base model name by removing probe type and layer suffixes.

    Returns
    -------
    str
        The base model name with suffixes removed.
    """
    # Remove common suffixes: _attention_last, _attention_all, _linear_last, _linear_all
    suffixes_to_remove = [
        "_attention_last",
        "_attention_all",
        "_linear_last",
        "_linear_all",
    ]
    base_name = full_name
    for suffix in suffixes_to_remove:
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break
    return base_name


def get_ssl_sl_classification(base_model_name: str) -> str:
    """Determine SSL/SL classification based on base model name.

    Returns
    -------
    str
        "SSL" or "SL" based on the model name.
    """
    name_lower = base_model_name.lower()

    # Special cases first
    if "bird_aves" in name_lower or "aves" in name_lower:
        return "SSL"
    if "beats_naturelm" in name_lower or "naturelm" in name_lower:
        return "SSL"
    if "efficientnet" in name_lower:
        return "SL"

    # Check for sl_ (but not ssl_)
    if name_lower.startswith("sl_") and not name_lower.startswith("ssl_"):
        return "SL"

    # Check for ssl_ (but not sl_)
    if "ssl_" in name_lower and not name_lower.startswith("sl_"):
        return "SSL"

    # Default to SL for other cases
    return "SL"


def parse_layer_weights(weights_str: str) -> List[float]:
    """
    Parse layer weights string into a list of floats.

    Parameters
    ----------
    weights_str : str
        Comma-separated string of weights, or 'nan' if missing.

    Returns
    -------
    List[float]
        List of parsed weights, or empty list if invalid.
    """
    if pd.isna(weights_str) or weights_str == "nan" or weights_str == "":
        return []

    try:
        weights = [float(w.strip()) for w in str(weights_str).split(",")]
        return weights
    except (ValueError, AttributeError):
        return []


def extract_layer_weights_data(csv_path: str) -> pd.DataFrame:
    """
    Extract layer weights data from CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the extracted metrics CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: base_model, probe_type, layers, dataset_name,
        layer_weights, layer_indices
    """
    df = pd.read_csv(csv_path)

    # Filter out rows with missing layer weights
    df = df[df["layer_weights"].notna() & (df["layer_weights"] != "nan")]

    # Parse layer weights
    df["parsed_weights"] = df["layer_weights"].apply(parse_layer_weights)

    # Filter out rows with empty parsed weights
    df = df[df["parsed_weights"].apply(len) > 0]

    # Create expanded rows for each layer
    expanded_rows = []
    for _, row in df.iterrows():
        weights = row["parsed_weights"]
        for i, weight in enumerate(weights):
            expanded_rows.append(
                {
                    "base_model": row["base_model"],
                    "probe_type": row["probe_type"],
                    "layers": row["layers"],
                    "dataset_name": row["dataset_name"],
                    "layer_index": i,
                    "weight": weight,
                }
            )

    return pd.DataFrame(expanded_rows)


def compute_error_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and error intervals for each base model and layer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with layer weights data.

    Returns
    -------
    pd.DataFrame
        DataFrame with mean, std, and confidence intervals.
    """
    # Group by base_model and layer_index, compute statistics
    stats = df.groupby(["base_model", "layer_index"]).agg({"weight": ["mean", "std", "count"]}).reset_index()

    # Flatten column names
    stats.columns = ["base_model", "layer_index", "mean", "std", "count"]

    # Compute 95% confidence interval (assuming normal distribution)
    # For small samples, use t-distribution approximation
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])  # Standard error of mean
    stats["ci_95"] = 1.96 * stats["sem"]  # 95% CI (approximate)

    # Add probe_type and layers info
    probe_info = df[["base_model", "probe_type", "layers"]].drop_duplicates()
    stats = stats.merge(probe_info, on="base_model", how="left")

    return stats


def plot_layer_weights(stats_df: pd.DataFrame, output_path: str, title: str) -> None:
    """
    Create line plots for layer weights with error intervals.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with statistics for plotting.
    output_path : str
        Path to save the plot.
    title : str
        Title for the plot.
    """
    # Set up the plot style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique base models
    base_models = stats_df["base_model"].unique()

    # Plot each base model
    for _, base_model in enumerate(base_models):
        model_data = stats_df[stats_df["base_model"] == base_model].sort_values("layer_index")

        if len(model_data) == 0:
            continue

        # Get probe info for labeling
        probe_type = model_data["probe_type"].iloc[0]
        layers = model_data["layers"].iloc[0]

        # Create label
        label = f"{base_model} ({probe_type}, {layers})"

        # Plot line with error bars
        ax.errorbar(
            model_data["layer_index"],
            model_data["mean"],
            yerr=model_data["ci_95"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=3,
            capthick=1,
            label=label,
            alpha=0.8,
        )

    # Customize plot
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Layer Weight", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved layer weights plot to {output_path}")


def plot_combined_layer_weights(beans_stats: pd.DataFrame, birdset_stats: pd.DataFrame, output_path: str) -> None:
    """
    Create combined plot showing both beans and birdset layer weights side by side.

    Parameters
    ----------
    beans_stats : pd.DataFrame
        Statistics for beans data.
    birdset_stats : pd.DataFrame
        Statistics for birdset data.
    output_path : str
        Path to save the combined plot.
    """
    # Set up the plot style
    plt.style.use("default")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot beans data
    beans_models = beans_stats["base_model"].unique()
    for _, base_model in enumerate(beans_models):
        model_data = beans_stats[beans_stats["base_model"] == base_model].sort_values("layer_index")
        if len(model_data) == 0:
            continue

        probe_type = model_data["probe_type"].iloc[0]
        layers = model_data["layers"].iloc[0]
        label = f"{base_model} ({probe_type}, {layers})"

        ax1.errorbar(
            model_data["layer_index"],
            model_data["mean"],
            yerr=model_data["ci_95"],
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            label=label,
            alpha=0.7,
        )

    ax1.set_xlabel("Layer Index", fontsize=12)
    ax1.set_ylabel("Layer Weight", fontsize=12)
    ax1.set_title("Beans Dataset - Layer Weights", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)

    # Plot birdset data
    birdset_models = birdset_stats["base_model"].unique()
    for _, base_model in enumerate(birdset_models):
        model_data = birdset_stats[birdset_stats["base_model"] == base_model].sort_values("layer_index")
        if len(model_data) == 0:
            continue

        probe_type = model_data["probe_type"].iloc[0]
        layers = model_data["layers"].iloc[0]
        label = f"{base_model} ({probe_type}, {layers})"

        ax2.errorbar(
            model_data["layer_index"],
            model_data["mean"],
            yerr=model_data["ci_95"],
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            label=label,
            alpha=0.7,
        )

    ax2.set_xlabel("Layer Index", fontsize=12)
    ax2.set_ylabel("Layer Weight", fontsize=12)
    ax2.set_title("Birdset Dataset - Layer Weights", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved combined layer weights plot to {output_path}")


def plot_averaged_layer_weights(beans_data: pd.DataFrame, birdset_data: pd.DataFrame, output_path: str) -> None:
    """
    Create a single plot averaging layer weights across both beans and birdset datasets.

    This function combines raw data from both datasets and computes statistics once,
    avoiding double-averaging that would occur if we averaged each dataset separately
    first.

    Parameters
    ----------
    beans_data : pd.DataFrame
        Raw layer weights data for beans.
    birdset_data : pd.DataFrame
        Raw layer weights data for birdset.
    output_path : str
        Path to save the averaged plot.
    """
    # Combine both datasets at the raw level
    combined_data = pd.concat(
        [beans_data.assign(dataset="beans"), birdset_data.assign(dataset="birdset")],
        ignore_index=True,
    )

    print(f"Combined raw data: {len(combined_data)} individual weight measurements")
    print(f"From {combined_data['base_model'].nunique()} unique base models")
    print(f"Across {combined_data['dataset'].nunique()} datasets: {combined_data['dataset'].unique()}")

    # Compute statistics directly from combined raw data
    # This is the correct approach - no double averaging
    stats = combined_data.groupby(["base_model", "layer_index"]).agg({"weight": ["mean", "std", "count"]}).reset_index()

    # Flatten column names
    stats.columns = ["base_model", "layer_index", "mean", "std", "count"]

    # Compute 95% confidence interval from the combined raw data
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci_95"] = 1.96 * stats["sem"]

    # Add probe_type and layers info
    probe_info = combined_data[["base_model", "probe_type", "layers"]].drop_duplicates()
    stats = stats.merge(probe_info, on="base_model", how="left")

    print(f"Statistics computed from {stats['count'].sum()} total measurements")
    print(f"Average measurements per layer: {stats['count'].mean():.1f}")

    # Set up the plot style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create mapping for base model names
    model_name_mapping = {
        "beats_naturelm": "NatureBEATs",
        "eat_hf_finetuned": "EAT",
        "eat_hf_pretrained": "EAT",
        "efficientnet_animalspeak_audioset": "EfficientNet",
        "sl_beats_all": "BEATs",
        "beats_pretrained": "BEATs",
        "bird_aves_bio": "AVES",
        "sl_eat_all_ssl_all": "EATall",
        "ssl_eat_all": "EATall",
    }

    # Apply the mapping to create display names
    combined_data_display = combined_data.copy()

    # Use the global functions for extracting base model name and SSL/SL classification

    # Extract base model names and apply mapping
    combined_data_display["base_model_clean"] = combined_data_display["base_model"].apply(extract_base_model_name)
    combined_data_display["display_name"] = combined_data_display["base_model_clean"].map(model_name_mapping)
    # Fill any unmapped names with the cleaned base model name
    combined_data_display["display_name"] = combined_data_display["display_name"].fillna(
        combined_data_display["base_model_clean"]
    )

    # Add SSL/SL classification
    combined_data_display["ssl_sl_class"] = combined_data_display["base_model_clean"].apply(get_ssl_sl_classification)
    combined_data_display["display_name_with_class"] = (
        combined_data_display["display_name"] + " (" + combined_data_display["ssl_sl_class"] + ")"
    )

    # Use seaborn lineplot directly on the raw combined data
    # This will automatically compute means and error bands
    sns.lineplot(
        data=combined_data_display,
        x="layer_index",
        y="weight",
        hue="display_name_with_class",
        errorbar="sd",  # Use standard deviation for error bands
        ax=ax,
        linewidth=2,
        markersize=5,
        alpha=0.8,
    )

    # Customize plot
    ax.set_xlabel("Layer Index", fontsize=14)
    ax.set_ylabel("Layer Weight", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Apply seaborn styling
    sns.despine()

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved averaged layer weights plot to {output_path}")
    print(f"Final plot shows {combined_data['base_model'].nunique()} models with statistics from combined raw data")


def plot_dataset_specific_layer_weights(beans_data: pd.DataFrame, birdset_data: pd.DataFrame, output_path: str) -> None:
    """
    Create a plot showing layer weights for each individual subdataset.
    Each subdataset (e.g., dog_classification, bat_classification, etc.) is a
    separate line.

    Parameters
    ----------
    beans_data : pd.DataFrame
        Raw layer weights data for beans.
    birdset_data : pd.DataFrame
        Raw layer weights data for birdset.
    output_path : str
        Path to save the dataset-specific plot.
    """
    # Set up the plot style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Combine both datasets with subdataset labels
    combined_data = pd.concat(
        [
            beans_data.assign(subdataset="Beans"),
            birdset_data.assign(subdataset="Birdset"),
        ],
        ignore_index=True,
    )

    # Extract individual subdatasets from the dataset_name column
    # The dataset_name column contains the actual subdataset names
    combined_data["subdataset"] = combined_data["dataset_name"]

    # Exclude EfficientNet from this plot (but keep it in model comparison plots)
    combined_data = combined_data[~combined_data["base_model"].str.contains("efficientnet", case=False, na=False)]

    print("After excluding EfficientNet:")
    print(f"  Remaining measurements: {len(combined_data)}")

    print("Combined data for subdataset visualization:")
    print(f"  Beans: {len(beans_data)} measurements")
    print(f"  Birdset: {len(birdset_data)} measurements")
    print(f"  Total: {len(combined_data)} measurements")

    # Get unique subdatasets
    unique_subdatasets = combined_data["subdataset"].unique()
    print(f"  Found {len(unique_subdatasets)} subdatasets: {sorted(unique_subdatasets)}")

    # Plot each subdataset as a separate line
    sns.lineplot(
        data=combined_data,
        x="layer_index",
        y="weight",
        hue="subdataset",
        errorbar="sd",  # Use standard deviation for error bands
        ax=ax,
        linewidth=2,
        markersize=4,
        alpha=0.8,
    )

    # Customize plot
    ax.set_xlabel("Layer Index", fontsize=14)
    ax.set_ylabel("Layer Weight", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10, framealpha=0.9)
    sns.despine()

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved subdataset-specific layer weights plot to {output_path}")
    print("Shows layer weights for each individual subdataset across all models")


def plot_layer_weights_heatmap(beans_data: pd.DataFrame, birdset_data: pd.DataFrame, output_path: str) -> None:
    """
    Create a heatmap of average layer weights per base model (rows) across
    layers (columns). Excludes EAT HF variants as requested.

    Parameters
    ----------
    beans_data : pd.DataFrame
        Raw layer weights data for beans.
    birdset_data : pd.DataFrame
        Raw layer weights data for birdset.
    output_path : str
        Path to save the heatmap plot.
    """
    # Combine raw data from both datasets
    combined = pd.concat([beans_data, birdset_data], ignore_index=True)

    # Exclude specified base models from the heatmap only
    mask_exclude = combined["base_model"].str.contains("eat_hf_pretrained", case=False, na=False) | combined[
        "base_model"
    ].str.contains("eat_hf_finetuned", case=False, na=False)
    filtered = combined[~mask_exclude].copy()

    if filtered.empty:
        print("No data left after excluding EAT HF variants; skipping heatmap.")
        return

    # Compute mean weight per base model and layer
    summary = filtered.groupby(["base_model", "layer_index"], as_index=False)["weight"].mean()

    # Pivot for heatmap (rows: base_model, columns: layer_index)
    heatmap_df = summary.pivot(index="base_model", columns="layer_index", values="weight")

    # Order base models alphabetically for readability
    heatmap_df = heatmap_df.sort_index()

    # Plot heatmap
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14, max(6, 0.5 * len(heatmap_df))))
    sns.heatmap(
        heatmap_df,
        cmap="viridis",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Average Layer Weight"},
        ax=ax,
    )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Base Model", fontsize=12)
    ax.set_title(
        "Layer Weights Heatmap (excluding EAT HF variants)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved layer weights heatmap to {output_path}")


def plot_dataset_specific_heatmap(beans_data: pd.DataFrame, birdset_data: pd.DataFrame, output_path: str) -> None:
    """
    Create a heatmap showing layer weights for each individual subdataset.
    Each subdataset is a separate row, aggregated across all models.
    Excludes EfficientNet as in the line plot version.

    Parameters
    ----------
    beans_data : pd.DataFrame
        Raw layer weights data for beans.
    birdset_data : pd.DataFrame
        Raw layer weights data for birdset.
    output_path : str
        Path to save the dataset-specific heatmap.
    """
    # Combine both datasets
    combined_data = pd.concat(
        [
            beans_data.assign(subdataset="Beans"),
            birdset_data.assign(subdataset="Birdset"),
        ],
        ignore_index=True,
    )

    # Extract individual subdatasets from the dataset_name column
    combined_data["subdataset"] = combined_data["dataset_name"]

    # Exclude EfficientNet from this plot (same as line plot version)
    combined_data = combined_data[~combined_data["base_model"].str.contains("efficientnet", case=False, na=False)]

    if combined_data.empty:
        print("No data left after excluding EfficientNet; skipping dataset heatmap.")
        return

    # Extract base model name and determine SSL/SL classification
    combined_data["base_model_clean"] = combined_data["base_model"].apply(extract_base_model_name)
    combined_data["ssl_sl"] = combined_data["base_model_clean"].apply(get_ssl_sl_classification)

    # Compute mean weight per subdataset, layer, and SSL/SL type
    summary = combined_data.groupby(["subdataset", "layer_index", "ssl_sl"], as_index=False)["weight"].mean()

    # Separate SSL and SL data
    ssl_data = summary[summary["ssl_sl"] == "SSL"].copy()
    sl_data = summary[summary["ssl_sl"] == "SL"].copy()

    # Define taxonomic ordering
    taxonomic_order = [
        # Birdset datasets
        "birdset_hsn_detection",
        "birdset_nbp_detection",
        "birdset_nes_detection",
        "birdset_per_detection",
        "birdset_pow_detection",
        "birdset_sne_detection",
        "birdset_uhh_detection",
        # Bird classification
        "bird_classification",
        # Enabirds detection
        "enabirds_detection",
        # RFCX detection
        "rfcx_detection",
        # DCASE detection
        "dcase_detection",
        # Mammal datasets
        "dog_classification",
        "hiceas_detection",
        "marine_mammal_classification",
        "hainan_gibbons_detection",
        "bat_classification",
        # Mosquito
        "mosquito_classification",
        # ESC50
        "esc50_classification",
    ]

    # Create display names with line breaks and replace underscores with spaces
    def format_dataset_name(name: str) -> str:
        # Replace all underscores with spaces
        name_spaced = name.replace("_", " ")
        if name_spaced.endswith(" classification"):
            return name_spaced.replace(" classification", "\n classification")
        elif name_spaced.endswith(" detection"):
            return name_spaced.replace(" detection", "\n detection")
        return name_spaced

    def normalize_to_10_layers(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize layer weights to exactly 10 columns.

        Redistributing first layer equally across all layers.

        Returns
        -------
        pd.DataFrame
            DataFrame with exactly 10 columns.
        """
        if df.empty:
            return df

        # If we have more than 10 columns, redistribute the first layer equally
        # across all 10 layers
        if len(df.columns) > 10:
            # Get the first layer weights (index 0)
            first_layer_weights = df.iloc[:, 0].values

            # Take columns 1-10 (skip the first column)
            df_10 = df.iloc[:, 1:11].copy()

            # Distribute first layer weights equally across all 10 layers
            equal_weight = 1.0 / 10  # Each layer gets 1/10 of the first layer's weight

            for i, weight in enumerate(first_layer_weights):
                if not pd.isna(weight) and weight != 0:
                    for j in range(10):
                        df_10.iloc[i, j] += weight * equal_weight

        elif len(df.columns) < 10:
            # If we have fewer than 10 columns, pad with zeros
            for i in range(len(df.columns), 10):
                df[f"layer_{i}"] = 0
            df_10 = df
        else:
            # If we have exactly 10 columns, use as is
            df_10 = df

        return df_10

    # Process SSL data
    if not ssl_data.empty:
        ssl_heatmap_df = ssl_data.pivot(index="subdataset", columns="layer_index", values="weight")
        available_ssl_datasets = [d for d in taxonomic_order if d in ssl_heatmap_df.index]
        ssl_heatmap_df = ssl_heatmap_df.reindex(available_ssl_datasets)
        ssl_heatmap_df = normalize_to_10_layers(ssl_heatmap_df)
        ssl_display_names = [format_dataset_name(name) for name in ssl_heatmap_df.index]
        ssl_heatmap_df.index = ssl_display_names
    else:
        ssl_heatmap_df = pd.DataFrame()

    # Process SL data
    if not sl_data.empty:
        sl_heatmap_df = sl_data.pivot(index="subdataset", columns="layer_index", values="weight")
        available_sl_datasets = [d for d in taxonomic_order if d in sl_heatmap_df.index]
        sl_heatmap_df = sl_heatmap_df.reindex(available_sl_datasets)
        sl_heatmap_df = normalize_to_10_layers(sl_heatmap_df)
        sl_display_names = [format_dataset_name(name) for name in sl_heatmap_df.index]
        sl_heatmap_df.index = sl_display_names
    else:
        sl_heatmap_df = pd.DataFrame()

    # Combine SSL and SL data for plotting
    if not ssl_heatmap_df.empty and not sl_heatmap_df.empty:
        # Ensure both have the same columns (layer indices)
        all_columns = ssl_heatmap_df.columns.union(sl_heatmap_df.columns)
        ssl_heatmap_df = ssl_heatmap_df.reindex(columns=all_columns, fill_value=0)
        sl_heatmap_df = sl_heatmap_df.reindex(columns=all_columns, fill_value=0)
        heatmap_df_display = pd.concat([ssl_heatmap_df, sl_heatmap_df])
    elif not ssl_heatmap_df.empty:
        heatmap_df_display = ssl_heatmap_df
    elif not sl_heatmap_df.empty:
        heatmap_df_display = sl_heatmap_df
    else:
        print("No data available for heatmap")
        return

    # Plot heatmap
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(16, max(10, 0.5 * len(heatmap_df_display))))
    sns.heatmap(
        heatmap_df_display,
        cmap="viridis",
        linewidths=0.3,
        linecolor="white",
        cbar=False,  # Remove colorbar legend
        annot=True,  # Show values in cells
        fmt=".2f",  # Format values to 2 decimal places
        annot_kws={"size": 16},  # Font size for cell annotations
        ax=ax,
    )

    # Add separator line between SSL and SL sections
    if not ssl_heatmap_df.empty and not sl_heatmap_df.empty:
        separator_y = len(ssl_heatmap_df)
        ax.axhline(y=separator_y, color="red", linewidth=2, alpha=0.7)

        # Add SSL/SL labels on the right side
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([])

        # Add SSL label for the top section
        if not ssl_heatmap_df.empty:
            ssl_mid_y = len(ssl_heatmap_df) / 2
            ax2.text(
                1.02,
                ssl_mid_y,
                "SSL",
                transform=ax2.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=16,
                fontweight="bold",
            )

        # Add SL label for the bottom section
        if not sl_heatmap_df.empty:
            sl_mid_y = len(ssl_heatmap_df) + len(sl_heatmap_df) / 2
            ax2.text(
                1.02,
                sl_mid_y,
                "SL",
                transform=ax2.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=16,
                fontweight="bold",
            )

    ax.set_xlabel("Layer Index", fontsize=20)
    # Remove ylabel completely
    ax.set_ylabel("", fontsize=20)

    # Increase tick label font sizes and make y-axis labels bold
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=12)

    # Make y-axis labels (subdatasets) bold
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved dataset-specific heatmap to {output_path}")


def plot_ssl_sl_heatmap(beans_csv: str, birdset_csv: str, output_path: str) -> None:
    """Create a heatmap separating SSL and SL datasets with a line."""
    # Read raw CSV data
    beans_df = pd.read_csv(beans_csv)
    birdset_df = pd.read_csv(birdset_csv)

    # Combine data from both datasets
    combined_data = pd.concat([beans_df, birdset_df], ignore_index=True)

    # Filter out rows with missing layer weights
    combined_data = combined_data[combined_data["layer_weights"].notna() & (combined_data["layer_weights"] != "nan")]

    # Extract base model name and determine SSL/SL classification
    combined_data["base_model_clean"] = combined_data["base_model"].apply(extract_base_model_name)
    combined_data["ssl_sl"] = combined_data["base_model_clean"].apply(get_ssl_sl_classification)

    # Parse layer weights and create long format
    layer_data = []
    for _, row in combined_data.iterrows():
        weights = parse_layer_weights(row["layer_weights"])
        if weights:
            for layer_idx, weight in enumerate(weights):
                layer_data.append(
                    {
                        "base_model": row["base_model_clean"],
                        "ssl_sl": row["ssl_sl"],
                        "layer_idx": layer_idx,
                        "weight": weight,
                    }
                )

    if not layer_data:
        print("No layer weights data found for SSL/SL heatmap")
        return

    df = pd.DataFrame(layer_data)

    # Compute average weights per base model and layer
    avg_weights = df.groupby(["base_model", "ssl_sl", "layer_idx"])["weight"].mean().reset_index()

    # Pivot to wide format for heatmap
    heatmap_data = avg_weights.pivot_table(
        index=["base_model", "ssl_sl"],
        columns="layer_idx",
        values="weight",
        fill_value=0,
    )

    # Separate SSL and SL data
    ssl_data = heatmap_data[heatmap_data.index.get_level_values("ssl_sl") == "SSL"]
    sl_data = heatmap_data[heatmap_data.index.get_level_values("ssl_sl") == "SL"]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [len(ssl_data), len(sl_data)]},
    )

    # Plot SSL data (top)
    if not ssl_data.empty:
        sns.heatmap(
            ssl_data.values,
            xticklabels=ssl_data.columns,
            yticklabels=[f"{name[0]} (SSL)" for name in ssl_data.index],
            cmap="viridis",
            linewidths=0.3,
            linecolor="white",
            cbar=True,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 12},
            ax=ax1,
        )
        ax1.set_title("SSL Models", fontsize=16, pad=20)
        ax1.set_xlabel("Layer Index", fontsize=14)
        ax1.set_ylabel("Base Model", fontsize=14)

    # Plot SL data (bottom)
    if not sl_data.empty:
        sns.heatmap(
            sl_data.values,
            xticklabels=sl_data.columns,
            yticklabels=[f"{name[0]} (SL)" for name in sl_data.index],
            cmap="viridis",
            linewidths=0.3,
            linecolor="white",
            cbar=True,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 12},
            ax=ax2,
        )
        ax2.set_title("SL Models", fontsize=16, pad=20)
        ax2.set_xlabel("Layer Index", fontsize=14)
        ax2.set_ylabel("Base Model", fontsize=14)

    # Add separator line between SSL and SL sections
    ax1.axhline(y=len(ssl_data), color="red", linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved SSL/SL heatmap to {output_path}")


def main() -> None:
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description="Plot layer weights with error intervals")
    parser.add_argument(
        "--beans-csv",
        default="evaluation_results/extracted_metrics_beans.csv",
        help="Path to beans CSV file",
    )
    parser.add_argument(
        "--birdset-csv",
        default="evaluation_results/extracted_metrics_birdset.csv",
        help="Path to birdset CSV file",
    )
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory for plots")
    parser.add_argument("--combined", action="store_true", help="Create combined plot for both datasets")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process beans data
    print("Processing beans data...")
    beans_data = extract_layer_weights_data(args.beans_csv)
    if len(beans_data) > 0:
        beans_stats = compute_error_intervals(beans_data)
        plot_layer_weights(
            beans_stats,
            os.path.join(args.output_dir, "layer_weights_beans.png"),
            "Beans Dataset - Layer Weights Across Models",
        )
        print(f"Beans: Found {len(beans_stats['base_model'].unique())} models with layer weights")
    else:
        print("No layer weights found in beans data")
        beans_stats = pd.DataFrame()

    # Process birdset data
    print("Processing birdset data...")
    birdset_data = extract_layer_weights_data(args.birdset_csv)
    if len(birdset_data) > 0:
        birdset_stats = compute_error_intervals(birdset_data)
        plot_layer_weights(
            birdset_stats,
            os.path.join(args.output_dir, "layer_weights_birdset.png"),
            "Birdset Dataset - Layer Weights Across Models",
        )
        print(f"Birdset: Found {len(birdset_stats['base_model'].unique())} models with layer weights")
    else:
        print("No layer weights found in birdset data")
        birdset_stats = pd.DataFrame()

    # Create combined plot if requested
    if args.combined and len(beans_stats) > 0 and len(birdset_stats) > 0:
        print("Creating combined plot...")
        plot_combined_layer_weights(
            beans_stats,
            birdset_stats,
            os.path.join(args.output_dir, "layer_weights_combined.png"),
        )

    # Create averaged plot (always create if both datasets have data)
    if len(beans_data) > 0 and len(birdset_data) > 0:
        print("Creating averaged plot...")
        plot_averaged_layer_weights(
            beans_data,
            birdset_data,
            os.path.join(args.output_dir, "layer_weights_averaged.png"),
        )

    # Create dataset-specific plot (always create if both datasets have data)
    if len(beans_data) > 0 and len(birdset_data) > 0:
        print("Creating dataset-specific plot...")
        plot_dataset_specific_layer_weights(
            beans_data,
            birdset_data,
            os.path.join(args.output_dir, "layer_weights_by_dataset.png"),
        )
        print("Creating layer weights heatmap...")
        plot_layer_weights_heatmap(
            beans_data,
            birdset_data,
            os.path.join(args.output_dir, "layer_weights_heatmap.png"),
        )
        print("Creating dataset-specific heatmap...")
        plot_dataset_specific_heatmap(
            beans_data,
            birdset_data,
            os.path.join(args.output_dir, "layer_weights_by_dataset_heatmap.png"),
        )
        print("Creating SSL/SL heatmap...")
        plot_ssl_sl_heatmap(
            args.beans_csv,
            args.birdset_csv,
            os.path.join(args.output_dir, "layer_weights_ssl_sl_heatmap.png"),
        )

    print("Layer weights plotting completed!")


if __name__ == "__main__":
    main()
