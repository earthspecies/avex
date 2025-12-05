#!/usr/bin/env bash

# Slurm job array for OpenBEATs experiments
# Usage: sbatch jobs/openbeats_ft_array.sh
#
# Runs multiple OpenBEATs model variants (uses GCS bucket directly)

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --output="/home/%u/logs/openbeats_%A_%a.log"
#SBATCH --job-name="openbeats-array"
#SBATCH --time=24:00:00
#SBATCH --array=1-2              # Run 2 experiments

# =============================================================================
# Define experiments - modify this array to add more configs
# =============================================================================
declare -A experiments=(
    [1]="openbeats-large-i2"     # Iteration 2
    [2]="openbeats-large-i3"     # Iteration 3
)

# Get current experiment
EXPERIMENT=${experiments[$SLURM_ARRAY_TASK_ID]}

if [ -z "$EXPERIMENT" ]; then
    echo "Error: No experiment defined for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== OpenBEATs Fine-tuning Array ==="
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Experiment: $EXPERIMENT"

source ~/slurm_env
cd ~/representation-learning

# Set up the virtual environment using uv
uv sync

export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Determine model size from experiment name
if [[ "$EXPERIMENT" == *"large"* ]]; then
    MODEL_SIZE="large"
elif [[ "$EXPERIMENT" == *"base"* ]]; then
    MODEL_SIZE="base"
else
    MODEL_SIZE="large"
fi

# Run training with experiment-specific model (uses GCS bucket directly)
srun uv run representation_learning/run_train.py \
    --config configs/run_configs/pretrained/openbeats_ft.yml \
    --patch model_spec.model_id="$EXPERIMENT" \
    --patch model_spec.model_size="$MODEL_SIZE" \
    --patch run_name="openbeats_ft_${EXPERIMENT}" \
    --patch output_dir="./runs/openbeats_ft_${EXPERIMENT}"

echo "=== Training Complete ==="
echo "End time: $(date)"
