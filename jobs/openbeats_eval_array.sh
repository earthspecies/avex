#!/usr/bin/env bash
# ==============================================================================
# OpenBEATs Evaluation Array Job (SLURM)
# ==============================================================================
# Runs all benchmarks from the paper in parallel using SLURM array jobs.
#
# Array task mapping:
#   1 = BEANS (Classification + Detection)
#   2 = BirdSet (Detection)
#   3 = Individual ID (Classification)
#   4 = Vocal Repertoire (Classification)
#
# Usage:
#   sbatch jobs/openbeats_eval_array.sh
# ==============================================================================

#SBATCH --array=1-4%4
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/openbeats_eval_%A_%a.log"
#SBATCH --job-name="openbeats-eval-array"
#SBATCH --cpus-per-gpu=12

set -e

# Map array task ID to config file
declare -A configs=(
    [1]="configs/evaluation_configs/openbeats/openbeats_beans.yml"
    [2]="configs/evaluation_configs/openbeats/openbeats_birdset.yml"
    [3]="configs/evaluation_configs/openbeats/openbeats_individual_id.yml"
    [4]="configs/evaluation_configs/openbeats/openbeats_vocal_repertoire.yml"
)

declare -A benchmark_names=(
    [1]="BEANS"
    [2]="BirdSet"
    [3]="Individual_ID"
    [4]="Vocal_Repertoire"
)

# Get the config file for this array task
config_file=${configs[$SLURM_ARRAY_TASK_ID]}
benchmark_name=${benchmark_names[$SLURM_ARRAY_TASK_ID]}

if [ -z "$config_file" ]; then
    echo "Error: No config file found for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=============================================="
echo "OpenBEATs Evaluation - $benchmark_name"
echo "=============================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Config: $config_file"
echo "Start time: $(date)"
echo ""

# Navigate to repository root
cd ~/representation-learning

# Environment setup
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export BEANS_DEBUG=0

# Sync dependencies
uv sync

# Run evaluation
srun uv run repr-learn evaluate --config "$config_file"

echo ""
echo "=============================================="
echo "Completed: $benchmark_name"
echo "End time: $(date)"
echo "=============================================="
