# Extract Metrics from Logs Script

This script extracts computed metrics from log files that run base model + probe experiments and generates a CSV file with each dataset result as a separate row.

## Usage

```bash
python scripts/extract_metrics_from_logs.py <log_file> [options]
```

### Arguments

- `log_file`: Path to the log file to process (required)

### Options

- `--output`, `-o`: Output CSV file path (default: `extracted_metrics.csv`)
- `--verbose`, `-v`: Enable verbose output to show extracted entries

## Features

The script extracts the following information from log files:

### Required Information
- **Probe Type**: Type of probe used (e.g., `weighted_attention`, `weighted_linear`)
- **Layers**: Target layers used (e.g., `last_layer`, `all`)
- **Base Model**: Name of the base model
- **Benchmark**: Benchmark name
- **Dataset**: Dataset name
- **Metrics**: Either `mAP` or `accuracy` values

### Parameter Information
- **Probe Parameters**: Trainable and total parameter counts for the probe
- **Base Model Parameters**: Trainable and total parameter counts for the base model

### Layer Weights (when available)
- **Layer Weights**: Comma-separated normalized layer weights as a string

## Log Format Expected

The script expects log files with the following format:

```
2025-09-11 08:32:07 | INFO | run_finetune: Probe configuration: type=weighted_attention, layers=['last_layer'], aggregation=none, input_processing=sequence, training_mode=online
2025-09-11 08:32:08 | INFO | run_finetune:   Probe parameters: 2.47M trainable / 2.47M total
2025-09-11 08:32:08 | INFO | run_finetune:   Base model parameters: 0 trainable / 90010417 total
2025-09-11 08:32:10 | INFO | run_finetune: [birdset_pow_detection | birdset | sl_eat_all_ssl_all_attention_last]  probe-test: {'mAP': 0.23582476377487183} | retrieval: n/a | clustering: n/a
2025-09-11 08:32:15 | INFO | run_finetune: Learned Layer Weights:
==================================================
Layer           Raw Weight   Normalized   Percentage
--------------------------------------------------
all             -0.5796      0.0505       5.05        %
Layer_1         -0.4250      0.0590       5.90        %
Layer_2         -0.2852      0.0678       6.78        %
...
--------------------------------------------------
```

## Output CSV Format

The generated CSV file contains the following columns (in order):

1. **dataset_name**: Name of the dataset (e.g., "birdset_pow_detection")
2. **probe_type**: Type of probe used (e.g., "weighted_attention")
3. **layers**: Target layers (e.g., "last_layer", "all")
4. **base_model**: Base model name
5. **benchmark**: Benchmark name
6. **probe_trainable**: Probe trainable parameters
7. **probe_total**: Probe total parameters
8. **base_trainable**: Base model trainable parameters
9. **base_total**: Base model total parameters
10. **layer_weights**: Comma-separated normalized layer weights
11. **metric**: The metric value (mAP or accuracy)

Each row represents one dataset result, making it easy to analyze and compare results across different datasets and configurations.

## Examples

### Basic usage:
```bash
python scripts/extract_metrics_from_logs.py experiment.log
```

### With custom output file:
```bash
python scripts/extract_metrics_from_logs.py experiment.log --output results.csv
```

### With verbose output:
```bash
python scripts/extract_metrics_from_logs.py experiment.log --verbose
```

## Notes

- The script will create the CSV file if it doesn't exist
- If the CSV file already exists, new results will be appended to it
- The script handles both `mAP` and `accuracy` metrics
- Layer weights are only extracted when the "Learned Layer Weights" section is present in the logs
- The script is robust to missing information and will fill empty values appropriately
