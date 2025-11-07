#!/usr/bin/env python3
"""
Script to add probe experiments to existing config files.

This script adds probe configurations to the experiments list in each config file
while preserving all existing experiments and structure.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML file and return its contents.

    Returns:
        Dict[str, Any]: The contents of the YAML file.
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """Save data to YAML file with proper formatting and newlines."""
    # Get base name from file path
    base_name = os.path.basename(file_path).replace(".yml", "")

    with open(file_path, "w") as f:
        # Write the header comment
        f.write("# On which datasets to run the evaluation and which metrics to use\n")
        f.write("dataset_config: " + str(data.get("dataset_config", "")) + "\n\n")

        # Write training_params
        f.write("training_params:\n")
        training_params = data.get("training_params", {})
        for key, value in training_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        # Write experiments
        f.write("# Which experiments to evaluate\n")
        f.write("experiments:\n")
        experiments = data.get("experiments", [])
        for i, exp in enumerate(experiments):
            f.write("  - run_name: " + str(exp.get("run_name", "")) + "\n")

            if "run_config" in exp:
                f.write("    run_config: " + str(exp["run_config"]) + "\n")

            if "probe_config" in exp:
                f.write("    probe_config:\n")
                probe_config = exp["probe_config"]
                for key, value in probe_config.items():
                    if isinstance(value, list):
                        # Format lists as inline YAML arrays
                        list_str = "[" + ", ".join(f'"{item}"' for item in value) + "]"
                        f.write(f"      {key}: {list_str}\n")
                    else:
                        f.write(f"      {key}: {value}\n")

            if "pretrained" in exp:
                f.write("    pretrained: " + str(exp["pretrained"]).lower() + "\n")

            if "checkpoint_path" in exp:
                f.write("    checkpoint_path: " + str(exp["checkpoint_path"]) + "\n")

            # Add newline between experiments except for the last one
            if i < len(experiments) - 1:
                f.write("\n")

        # Add commented-out experiments for lstm, mlp, transformer
        commented_probes = [
            ("lstm", "all", True),
            ("lstm", "last", True),
            ("lstm", "ft", False),
            ("mlp", "all", True),
            ("mlp", "last", True),
            ("mlp", "ft", False),
            ("transformer", "all", True),
            ("transformer", "last", True),
            ("transformer", "ft", False),
        ]

        for probe_type, layer_info, freeze_backbone in commented_probes:
            f.write(f"\n  # - run_name: {base_name}_{probe_type}_{layer_info}\n")
            f.write("  #   run_config: configs/run_configs/pretrained/beats_ft.yml\n")
            f.write("  #   probe_config:\n")
            f.write(f'  #     probe_type: "weighted_{probe_type}"\n')

            if probe_type in ["lstm", "transformer"]:
                f.write('  #     aggregation: "none"\n')
                f.write('  #     input_processing: "sequence"\n')
            else:  # mlp
                f.write('  #     aggregation: "mean"\n')
                f.write('  #     input_processing: "pooled"\n')

            if layer_info == "ft":
                f.write('  #     target_layers: ["last_layer"]\n')
            else:
                f.write(f'  #     target_layers: ["{layer_info}"]\n')

            if probe_type == "lstm":
                f.write("  #     lstm_hidden_size: 64\n")
                f.write("  #     num_layers: 1\n")
                f.write("  #     bidirectional: true\n")
                f.write("  #     max_sequence_length: 1000\n")
            elif probe_type == "transformer":
                f.write("  #     num_heads: 8\n")
                f.write("  #     attention_dim: 128\n")
                f.write("  #     num_layers: 1\n")
                f.write("  #     max_sequence_length: 1200\n")
            elif probe_type == "mlp":
                f.write("  #     hidden_dims: [512, 256]\n")
                f.write("  #     dropout_rate: 0.3\n")
                f.write('  #     activation: "gelu"\n')

            if probe_type in ["lstm", "transformer"]:
                f.write("  #     use_positional_encoding: false\n")
                f.write("  #     dropout_rate: 0.3\n")

            f.write(f"  #     freeze_backbone: {str(freeze_backbone).lower()}\n")
            f.write("  #     online_training: true\n")
            f.write("  #   pretrained: false\n")
            f.write("  #   checkpoint_path: gs://representation-learning/models/sl_beats_all.pt\n")

        f.write("\n")

        # Write save_dir
        f.write("# Where to save the evaluation results\n")
        f.write("save_dir: " + str(data.get("save_dir", "")) + "\n\n")

        # Write results_csv_path
        f.write("# Optional: Append results to a global CSV file for cross-model comparison\n")
        f.write("results_csv_path: " + str(data.get("results_csv_path", "")) + "\n\n")

        # Write device
        f.write("device: " + str(data.get("device", "")) + "\n\n")

        # Write seed
        f.write("seed: " + str(data.get("seed", "")) + "\n\n")

        # Write num_workers
        f.write(
            "num_workers: " + str(data.get("num_workers", "")) + "  # Enable workers for better memory management\n\n"
        )

        # Write eval_modes
        f.write("# Evaluation phases\n")
        f.write("eval_modes:\n")
        eval_modes = data.get("eval_modes", [])
        for mode in eval_modes:
            f.write("  - " + str(mode) + "\n")
        f.write("\n")

        # Write overwrite_embeddings
        f.write("overwrite_embeddings: " + str(data.get("overwrite_embeddings", "")).lower() + "\n\n")

        # Write disable_tqdm
        f.write("# Control tqdm progress bar verbosity during fine-tuning\n")
        f.write("# Set to true to disable progress bars and reduce output verbosity\n")
        f.write("disable_tqdm: " + str(data.get("disable_tqdm", "")).lower() + "\n")


def get_probe_configs() -> List[Dict[str, Any]]:
    """Get the probe configurations from the reference file.

    Returns:
        List[Dict[str, Any]]: List of probe configuration dictionaries.
    """
    reference_file = "configs/evaluation_configs/icassp/sl_beats_all.yml"

    if not os.path.exists(reference_file):
        print(f"Error: Reference file {reference_file} not found")
        return []

    data = load_yaml(reference_file)
    experiments = data.get("experiments", [])

    # Extract probe configurations (those with probe types in the name)
    # Only include attention and linear probes (exclude lstm, mlp, transformer)
    probe_configs = []
    for exp in experiments:
        run_name = exp.get("run_name", "")
        # Look for probe types in the run name
        if any(probe_type in run_name for probe_type in ["attention", "linear"]):
            probe_configs.append(exp)

    print(f"Found {len(probe_configs)} probe configurations from reference file")
    return probe_configs


def create_probe_experiment(
    original_run_name: str,
    probe_config: Dict[str, Any],
    original_experiment: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a new probe experiment based on the original run name, probe config,
    and original experiment.

    Returns:
        Dict[str, Any]: The new probe experiment configuration.
    """
    # Extract probe type and layer info from the reference run_name
    ref_run_name = probe_config["run_name"]
    has_ft_suffix = ref_run_name.endswith("_ft") or "_ft_" in ref_run_name
    probe_parts = ref_run_name.split("_")

    # Find the probe type and layer info
    probe_type = None
    layer_info: str | None = None

    for part in probe_parts:
        if part in ["transformer", "attention", "lstm", "linear", "mlp"]:
            probe_type = part
            break

    # Attempt to infer layer info from the run_name first
    if "last" in ref_run_name:
        layer_info = "last"
    elif "all" in ref_run_name:
        layer_info = "all"
    else:
        # Fallback: infer from the reference probe_config target_layers
        ref_probe_cfg = probe_config.get("probe_config", {})
        ref_target_layers = ref_probe_cfg.get("target_layers")
        if isinstance(ref_target_layers, list) and ref_target_layers:
            first_layer = ref_target_layers[0]
            if first_layer == "last_layer":
                layer_info = "last"
            elif first_layer == "all":
                layer_info = "all"

    if not probe_type:
        return None

    # Build new run name
    if has_ft_suffix:
        new_run_name = f"{original_run_name}_{probe_type}_ft"
    else:
        if not layer_info:
            return None
        new_run_name = f"{original_run_name}_{probe_type}_{layer_info}"

    # Create new experiment using original values for run_config, pretrained,
    # checkpoint_path
    new_experiment = {
        "run_name": new_run_name,
        "run_config": original_experiment.get("run_config", ""),
        "probe_config": probe_config["probe_config"].copy(),
        "pretrained": original_experiment.get("pretrained", False),
    }

    # Set correct target_layers based on layer_info or ft variant (ft uses
    # last_layer per reference)
    if has_ft_suffix:
        new_experiment["probe_config"]["target_layers"] = ["last_layer"]
    else:
        if layer_info == "last":
            new_experiment["probe_config"]["target_layers"] = ["last_layer"]
        elif layer_info == "all":
            new_experiment["probe_config"]["target_layers"] = ["all"]

    # Only add checkpoint_path if it exists in the original experiment
    if "checkpoint_path" in original_experiment:
        new_experiment["checkpoint_path"] = original_experiment["checkpoint_path"]

    return new_experiment


def add_probe_experiments_to_file(config_path: str, dry_run: bool = False) -> None:
    """Add probe experiments to a single config file."""
    print(f"Processing: {config_path}")

    # Load the config
    data = load_yaml(config_path)

    # Get the original experiments
    original_experiments = data.get("experiments", [])
    if not original_experiments:
        print(f"  Warning: No experiments found in {config_path}")
        return

    # Get probe configurations
    probe_configs = get_probe_configs()
    if not probe_configs:
        print("  Warning: No probe configurations found")
        return

    # Create new probe experiments using the first original experiment as template
    template_experiment = original_experiments[0]
    original_run_name = template_experiment["run_name"]
    new_experiments = []
    for probe_config in probe_configs:
        new_experiment = create_probe_experiment(original_run_name, probe_config, template_experiment)
        if new_experiment:
            new_experiments.append(new_experiment)

    # Sort new experiments by probe_type
    new_experiments.sort(key=lambda x: x.get("probe_config", {}).get("probe_type", ""))

    if not new_experiments:
        print(f"  Warning: No probe experiments created for {config_path}")
        return

    # Add the new experiments to the existing list
    existing_run_names = {exp.get("run_name") for exp in original_experiments}
    deduped_new_experiments = [exp for exp in new_experiments if exp.get("run_name") not in existing_run_names]
    all_experiments = original_experiments + deduped_new_experiments
    data["experiments"] = all_experiments

    if not dry_run:
        # Save the updated config
        save_yaml(data, config_path)
        print(f"  Added {len(deduped_new_experiments)} probe experiments (total: {len(all_experiments)})")
    else:
        print(f"  Would add {len(deduped_new_experiments)} probe experiments (total: {len(all_experiments)}):")
        for exp in deduped_new_experiments:
            print(f"    - {exp['run_name']}")


def main() -> None:
    """Main function to add probe experiments to all config files."""
    parser = argparse.ArgumentParser(description="Add probe experiments to icassp config files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/evaluation_configs/icassp",
        help="Directory containing config files",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)

    # Files to exclude from processing (reference files)
    exclude_files = {"sl_beats_all.yml", "sl_efficientnet_audioset.yml"}

    # Process all config files except the reference files
    config_files = [f for f in config_dir.glob("*.yml") if f.name not in exclude_files]

    print(f"Processing {len(config_files)} config files...")

    for config_file in sorted(config_files):
        try:
            add_probe_experiments_to_file(str(config_file), args.dry_run)
        except Exception as e:
            print(f"Error processing {config_file}: {e}")

    if args.dry_run:
        print("\nDry run completed. Use without --dry-run to apply changes.")
    else:
        print("\nAll config files have been updated successfully.")


if __name__ == "__main__":
    main()
