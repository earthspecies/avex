#!/usr/bin/env python3
"""
Script to create simplified summaries from evaluation result JSONL files.

This script reads specified JSONL files, extracts only the columns we want to keep,
and saves the simplified results with a '_simple' postfix.
"""

import json
import os
from pathlib import Path


def process_jsonl_file(
    input_file: str, columns_to_keep: list[str]
) -> tuple[str, list[dict]]:
    """
    Process a JSONL file to keep only specified columns.

    Args:
        input_file: Path to the input JSONL file
        columns_to_keep: List of column names to preserve

    Returns:
        Tuple of (output_file_path, processed_data_list)
    """
    input_path = Path(input_file)

    # Create output filename with _simple postfix
    output_file = input_path.with_name(input_path.stem + "_simple" + input_path.suffix)

    print(f"Processing {input_file} -> {output_file}")

    if not input_path.exists():
        print(f"WARNING: Input file {input_file} does not exist, skipping...")
        return str(output_file), []

    processed_lines = []

    with open(input_path, "r") as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Extract only the columns we want to keep
                simplified_data = {}
                for column in columns_to_keep:
                    if column in data:
                        simplified_data[column] = data[column]
                    else:
                        # Set to null if column doesn't exist
                        simplified_data[column] = None

                processed_lines.append(simplified_data)

            except json.JSONDecodeError as e:
                print(
                    f"ERROR: Could not parse JSON on line {line_num} in {input_file}: {e}"
                )
                continue

    # Write the simplified data
    with open(output_file, "w") as outfile:
        for data in processed_lines:
            json.dump(data, outfile, separators=(",", ":"))
            outfile.write("\n")

    print(f"Wrote {len(processed_lines)} lines to {output_file}")
    return str(output_file), processed_lines


def main():
    """Main function to process all specified JSONL files."""

    # Hard-coded list of JSONL files to process
    jsonl_files = [
        # "evaluation_results/efficientnet_beans/summary_2025-07-18 03:02:32.065485.jsonl",
        # "evaluation_results/efficientnet_beans/summary_2025-07-18 03:02:32.673098.jsonl",
        # "evaluation_results/aaai/eat_audioset/summary_2025-07-18 01:47:06.311017.jsonl",
        # "evaluation_results/aaai/eat_all/summary_2025-07-18 03:20:59.285709.jsonl",
        # "evaluation_results/aaai/beats_raw/summary_2025-07-18 05:17:32.301897.jsonl",
        # "evaluation_results/aaai/beats_ft/summary_2025-07-18 05:16:12.360611.jsonl",
        # "evaluation_results/aaai/bird_aves_bio/summary_2025-07-18 05:05:48.646055.jsonl",
        # "evaluation_results/aaai/eat_audioset/summary_2025-07-18 04:40:24.685877.jsonl",
        # "evaluation_results/efficientnet_beans/summary_2025-07-18 03:02:32.673098.jsonl",
        # "evaluation_results/aaai/beats_raw/summary_2025-07-19 02:54:31.352008.jsonl",
        # "evaluation_results/aaai/bird_aves_bio/summary_2025-07-19 02:24:47.161734.jsonl",
        # "evaluation_results/aaai/eat_all/summary_2025-07-18 21:53:21.655903.jsonl", # EAT ALL probe
        # "evaluation_results/aaai/eat_audioset/summary_2025-07-18 17:50:51.267236.jsonl",
        # "evaluation_results/efficientnet_beans/summary_2025-07-18 21:42:30.458457.jsonl", # EfficientNet AnimalSpeak probe
        # "evaluation_results/atst_frame/summary_2025-07-18 21:37:14.555761.jsonl", # Atst Frame probe
        # "evaluation_results/aaai/eat_audioset/summary_2025-07-20 14:15:06.823734.jsonl", # EAT Audioset probe
        # "evaluation_results/bird_aves_bio/summary_2025-07-20 15:30:46.762704.jsonl", #Perch probe BEANs
        # # "evaluation_results/aaai/sl_beats_animalspeak/summary_2025-07-20 14:51:14.551121.jsonl" # BEATs_sl_v0 probe?
        # "evaluation_results/aaai/beats_naturelm/summary_2025-07-20 19:43:19.374768.jsonl", # BEATs_NatureLM probe
        # "evaluation_results/aaai/eat_baseline_finetuned/summary_2025-07-20 21:30:17.202169.jsonl", # EAT baseline finetuned probe
        # "evaluation_results/aaai/eat_baseline/summary_2025-07-20 22:01:58.639929.jsonl", # EAT baseline probe
        # "evaluation_results/efficientnet_beans/summary_2025-07-20 21:56:01.313972.jsonl", # EfficientNet Audioset probe
        # "evaluation_results/aaai/sl_beats_animalspeak/summary_2025-07-20 15:25:44.410672.jsonl", # BEATs_sl_v0 probe?
        # "evaluation_results/aaai/clap_efficientnet_animalspeak/summary_2025-07-21 04:57:08.671140.jsonl", # CLAP-animalspeak probe
        # "evaluation_results/aaai/clap_efficientnet_animalspeak_audioset/summary_2025-07-21 04:57:42.115547.jsonl", # CLAP-animalspeak-audioset probe
        "evaluation_results/aaai/eat_all/summary_2025-07-25 03:51:58.291412.jsonl",  # EAT ALL indv
        "evaluation_results/aaai/eat_audioset/summary_2025-07-25 03:51:59.346372.jsonl",  # EAT AnimalSpeak indv
        "evaluation_results/aaai/eat_audioset/summary_2025-07-25 04:22:23.751059.jsonl",  # EAT Audioset indv
        "evaluation_results/beats/summary_2025-07-25 03:56:48.293576.jsonl",  # BEATs pretrain indv
        "evaluation_results/beats_naturelm/summary_2025-07-25 03:39:12.561771.jsonl",  # BEATs NatureLM indv
        "evaluation_results/bird_aves_bio/summary_2025-07-25 04:16:17.511071.jsonl",  # Prech indv
        "evaluation_results/aaai/efficientnet_audioset/summary_2025-07-25 04:32:38.411188.jsonl",  # EfficientNet Audioset indv
        "evaluation_results/aaai/eat_audioset/summary_2025-07-25 05:55:43.188113.jsonl",  # EAT hf-pre indv
        "evaluation_results/beats/summary_2025-07-25 06:00:19.952179.jsonl",  # BEATs ft indv
        "evaluation_results/beats/summary_2025-07-25 05:49:08.376837.jsonl",  # BEATs pretrain indv
        "evaluation_results/aaai/sl_beats_all/summary_2025-07-25 06:32:46.794025.jsonl",  # BEATs sl all indv
        "evaluation_results/aaai/sl_beats_animalspeak/summary_2025-07-25 06:32:40.833836.jsonl",  # sl BEATs animalspeak indv
        "evaluation_results/aaai/sl_eat_all_ssl_all/summary_2025-07-25 06:50:03.995096.jsonl",  # sl EAT all ssl all indv
        "evaluation_results/aaai/sl_eat_animalspeak_ssl_all/summary_2025-07-25 06:51:25.782279.jsonl",  # sl EAT animalspeak ssl all indv
        "evaluation_results/aaai/eat_hf_pretrained/summary_2025-07-25 07:08:43.364139.jsonl",  # EAT hf pretrained indv
        "evaluation_results/aaai/eat_hf_finetuned/summary_2025-07-25 07:10:27.236915.jsonl",  # EAT hf ft indv
        "evaluation_results/aaai/sl_beats_all/summary_2025-07-26 03:47:34.700307.jsonl",  # sl BEATs all BEANs
        "evaluation_results/aaai/sl_beats_animalspeak/summary_2025-07-26 03:46:27.350908.jsonl",  # sl BEATs animalspeak BEANs
        "evaluation_results/aaai/sl_eat_all_ssl_all/summary_2025-07-26 06:21:39.040478.jsonl",  # sl EAT all ssl all BEANs
        "evaluation_results/aaai/sl_eat_animalspeak_ssl_all/summary_2025-07-26 06:26:17.765777.jsonl",  # sl EAT animalspeak ssl all BEANs
        "evaluation_results/birdnet/summary_2025-07-26 12:32:28.502268.jsonl",  # BirdNet probe
        "evaluation_results/birdmae/summary_2025-07-26 08:33:19.703647.jsonl",  # BirdMae base retrieval
        "evaluation_results/birdnet/summary_2025-07-27 03:58:19.215249.jsonl",  # BirdNet BirdSet
        "evaluation_results/aaai/efficientnet_animalspeak_soundscape/summary_2025-07-27 04:54:40.496518.jsonl",  # EfficientNet Animalspeak Soundscape probe
        "evaluation_results/aaai/efficientnet_animalspeak_wabad/summary_2025-07-27 04:52:58.991726.jsonl",  # EfficientNet Animalspeak WABAD probe
        "evaluation_results/aaai/sl_eat_animalspeak_ssl_all/summary_2025-07-27 06:37:46.379737.jsonl",  # sl EAT animalspeak ssl all birdset
        "evaluation_results/aaai/efficientnet_audioset/summary_2025-07-27 06:33:39.644214.jsonl",  # EfficientNet Audioset birdset
        "evaluation_results/aaai/eat_audioset/summary_2025-07-27 06:45:31.285396.jsonl",  # EAT animalspeak birdset
        "evaluation_results/beats/summary_2025-07-27 04:07:50.415851.jsonl",  # BEATs finetune BEANS
        "evaluation_results/aaai/sl_eat_all_ssl_all/summary_2025-07-27 06:38:13.908581.jsonl",  # sl EAT all ssl all birdset
        "evaluation_results/bird_aves_bio/summary_2025-07-27 04:48:28.737623.jsonl",  # AVES indv
        "evaluation_results/consolidated_beans_models_part1/summary_2025-07-28 01:35:06.521972.jsonl",  # Consolidated Finch 1
        "evaluation_results/consolidated_beans_models_part2/summary_2025-07-28 02:03:05.687166.jsonl",  # Consolidated Finch 2
        "evaluation_results/consolidated_beans_models_part3/summary_2025-07-28 01:45:47.268183.jsonl",  # Consolidated Finch 3
        "evaluation_results/aaai/efficientnet_animalspeak/summary_2025-07-30 05:25:28.783173.jsonl",  # EfficientNet Animalspeak indv
        "evaluation_results/aaai/efficientnet_animalspeak_audioset/summary_2025-07-30 05:25:23.818181.jsonl",  # EfficientNet Animalspeak Audioset indv
    ]

    # Hard-coded list of columns to keep
    columns_to_keep = [
        "dataset_name",
        "experiment_name",
        "evaluation_dataset_name",
        "test_mAP",
        # "test_multiclass_f1",
        "test_accuracy",
        # "test_balanced_accuracy",
        # "accuracy",
        "retrieval_roc_auc",
        "retrieval_precision_at_1",
        "clustering_nmi",
        "clustering_ari",
    ]

    print(f"Processing {len(jsonl_files)} JSONL files...")
    print(f"Keeping columns: {', '.join(columns_to_keep)}")
    print()

    output_files = []
    all_concatenated_data = []

    for jsonl_file in jsonl_files:
        output_file, processed_data = process_jsonl_file(jsonl_file, columns_to_keep)
        output_files.append(output_file)
        all_concatenated_data.extend(processed_data)

    # Create concatenated JSONL file
    concatenated_file = "evaluation_results/all_simple_results_concatenated.jsonl"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(concatenated_file), exist_ok=True)

    with open(concatenated_file, "w") as outfile:
        for data in all_concatenated_data:
            json.dump(data, outfile, separators=(",", ":"))
            outfile.write("\n")

    print()
    print("Summary:")
    print(f"Processed {len(jsonl_files)} input files")
    print(
        f"Created concatenated file with {len(all_concatenated_data)} total records: {concatenated_file}"
    )
    print()
    print("Individual output files created:")
    for output_file in output_files:
        print(f"  - {output_file}")


if __name__ == "__main__":
    main()
