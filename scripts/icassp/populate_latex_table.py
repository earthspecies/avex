#!/usr/bin/env python3
"""
Script to populate LaTeX table with metrics from extracted_metrics_birdset.csv
"""

from typing import Dict, Tuple

import pandas as pd


def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load and process the CSV data.

    Returns:
        pd.DataFrame: Filtered dataframe with derived ``model_family`` column.
    """
    df = pd.read_csv(csv_path)

    # Filter for birdset data only
    df = df[df["benchmark"] == "birdset"]

    # Create model mapping based on base_model names
    def get_model_family(base_model: str) -> str:
        if "efficientnet" in base_model:
            return "EfficientNet"
        elif "beats" in base_model:
            return "BEATs"
        elif "eat" in base_model:
            return "EAT"
        elif "bird_aves_bio" in base_model:
            return "AVES"
        else:
            return "Unknown"

    df["model_family"] = df["base_model"].apply(get_model_family)

    return df


def calculate_parameter_counts(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Calculate parameter counts for each model family and configuration.

    Returns:
        Dict[str, Dict[str, str]]: Mapping from model family to configuration
        totals with ``probe_total`` and ``base_total``.
    """
    results = {}

    # Group by model family, probe type, and layers
    grouped = df.groupby(["model_family", "probe_type", "layers"])

    for (model_family, probe_type, layers), group in grouped:
        if model_family not in results:
            results[model_family] = {}

        # Get probe_total and base_total (should be consistent within each group)
        probe_total = group["probe_total"].iloc[0]
        base_total = group["base_total"].iloc[0]

        # Create key for this configuration
        probe_key = "attention" if "attention" in probe_type else "linear"
        layer_key = "last" if "last" in layers else "all"

        config_key = f"{probe_key}_{layer_key}"
        results[model_family][config_key] = {
            "probe_total": probe_total,
            "base_total": base_total,
        }

    return results


def format_parameter_count(value: str) -> str:
    """Format parameter count (already formatted as string like '105.14M').

    Returns:
        str: The formatted count string.
    """
    return value


def generate_latex_table(results: Dict[str, Dict[str, Dict[str, str]]]) -> str:
    """Generate the populated LaTeX table.

    Returns:
        str: LaTeX table text.
    """

    def get_parameters(model_name: str) -> Tuple[str, str, str, str, str]:
        if model_name not in results:
            return "---", "---", "---", "---", "---"

        model_data = results[model_name]

        # Get base_total from any configuration (should be the same)
        base_total = "---"
        for config_data in model_data.values():
            base_total = format_parameter_count(config_data["base_total"])
            break

        # Get probe totals for each configuration
        att_last = format_parameter_count(
            model_data.get("attention_last", {}).get("probe_total", "---")
        )
        att_all = format_parameter_count(
            model_data.get("attention_all", {}).get("probe_total", "---")
        )
        lin_last = format_parameter_count(
            model_data.get("linear_last", {}).get("probe_total", "---")
        )
        lin_all = format_parameter_count(
            model_data.get("linear_all", {}).get("probe_total", "---")
        )

        return base_total, att_last, att_all, lin_last, lin_all

    latex_lines = [
        "\\begin{table*}[ht]",
        "\\centering",
        "\\caption{Base models included in the benchmark. We compare self-supervised (SSL) and supervised (SL) models, including post-trained variants, across diverse data sources.}",  # noqa: E501
        "\\label{tab:base_models}",
        "\\begin{tabular}{llllllllll}",
        "\\toprule",
        "\\textbf{Model}           & \\textbf{Pre-training Data}            & \\textbf{Post-training Data} & \\textbf{N.P. base} & \\multicolumn{4}{c}{\\textbf{N.P. probe}} & \\\\",  # noqa: E501
        "\\cmidrule{5-8}",
        "& & & & \\multicolumn{2}{c}{\\textbf{Attention}} & \\multicolumn{2}{c}{\\textbf{Linear}} & \\\\",  # noqa: E501
        "\\cmidrule{5-6} \\cmidrule{7-8}",
        "& & & & \\textbf{Last} & \\textbf{All} & \\textbf{Last} & \\textbf{All} & \\\\",  # noqa: E501
        "\\midrule",
    ]

    # Pre-training section
    # BEATs
    base_total, att_last, att_all, lin_last, lin_all = get_parameters("BEATs")
    latex_lines.append(
        f"BEATs                                        & AudioSet + Speech                     & --    & {base_total} & {att_last} & {att_all} & {lin_last} & {lin_all} &                                      \\\\"  # noqa: E501
    )  # noqa: E501

    # EAT
    base_total, att_last, att_all, lin_last, lin_all = get_parameters("EAT")
    latex_lines.append(
        f"EAT                                          & AudioSet + Speech                     & --    & {base_total} & {att_last} & {att_all} & {lin_last} & {lin_all} &                                      \\\\"  # noqa: E501
    )  # noqa: E501

    # EATall (use EAT data)
    base_total, att_last, att_all, lin_last, lin_all = get_parameters("EAT")
    latex_lines.append(
        f"EATall                                          & Bio  + AudioSet                   & --    & {base_total} & {att_last} & {att_all} & {lin_last} & {lin_all} &                                       \\\\"  # noqa: E501
    )  # noqa: E501

    # AVES
    base_total, att_last, att_all, lin_last, lin_all = get_parameters("AVES")
    latex_lines.append(
        f"AVES                                        & Xeno Canto                      & --   & {base_total} & {att_last} & {att_all} & {lin_last} & {lin_all} &                                       \\\\"  # noqa: E501
    )  # noqa: E501

    latex_lines.append("\\midrule")

    # Post-training section
    # BEATs (post-trained)
    base_total, att_last, att_all, lin_last, lin_all = get_parameters("BEATs")
    latex_lines.append(
        f"BEATs                                        & AudioSet + Speech                     & Bio + AudioSet    & {base_total} & {att_last} & {att_all} & {lin_last} & {lin_all} &                                      \\\\"  # noqa: E501
    )  # noqa: E501

    # NatureBEATs (use BEATs data)
    base_total, att_last, att_all, lin_last, lin_all = get_parameters("BEATs")
    latex_lines.append(
        f"NatureBEATs                          & AudioSet + Speech                     & Bio + Text prompts   & {base_total} & {att_last} & {att_all} & {lin_last} & {lin_all} &                \\\\"  # noqa: E501
    )  # noqa: E501

    # EAT (post-trained)
    base_total, att_last, att_all, lin_last, lin_all = get_parameters("EAT")
    latex_lines.append(
        f"EAT                  & AudioSet + Speech    & Bio + AudioSet  & {base_total} & {att_last} & {att_all} & {lin_last} & {lin_all} &        \\\\"  # noqa: E501
    )  # noqa: E501

    # EATall (post-trained, use EAT data)
    base_total, att_last, att_all, lin_last, lin_all = get_parameters("EAT")
    latex_lines.append(
        f"EATall                  & AudioSet + Speech    & Bio + AudioSet  & {base_total} & {att_last} & {att_all} & {lin_last} & {lin_all} &        \\\\"  # noqa: E501
    )  # noqa: E501

    # EfficientNet
    base_total, att_last, att_all, lin_last, lin_all = get_parameters("EfficientNet")
    latex_lines.append(
        f"EfficientNet                               & ImageNet                               & Bio + AudioSet & {base_total} & {att_last} & {att_all} & {lin_last} & {lin_all} &      \\\\"  # noqa: E501
    )  # noqa: E501

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")

    return "\n".join(latex_lines)


def main() -> None:
    """Main function to process data and generate LaTeX table."""
    csv_path = "evaluation_results/extracted_metrics_birdset.csv"

    # Load and process data
    df = load_and_process_data(csv_path)

    # Calculate parameter counts
    results = calculate_parameter_counts(df)

    # Print results for debugging
    print("Extracted parameter counts by model family:")
    for model, configs in results.items():
        print(f"\n{model}:")
        for config, data in configs.items():
            print(
                f"  {config}: probe_total={data['probe_total']}, base_total={data['base_total']}"  # noqa: E501
            )

    # Generate LaTeX table
    latex_table = generate_latex_table(results)

    # Save to file
    with open("evaluation_results/populated_latex_table.tex", "w") as f:
        f.write(latex_table)

    print("\nLaTeX table saved to evaluation_results/populated_latex_table.tex")
    print("\nGenerated LaTeX table:")
    print(latex_table)


if __name__ == "__main__":
    main()
