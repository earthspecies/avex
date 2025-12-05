#!/usr/bin/env bash

# Slurm job script for OpenBEATs supervised fine-tuning on a single GPU
# Usage: sbatch jobs/openbeats_ft_single_gpu.sh

#SBATCH --partition=a100-40      # GPU partition - need larger GPU for OpenBEATs-Large
#SBATCH --gpus=1                  # Single GPU
#SBATCH --cpus-per-gpu=8          # CPU cores per GPU
#SBATCH --mem=64G                 # Memory (increased for large model)
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-openbeats-ft"
#SBATCH --time=24:00:00          # Max job duration

echo "=== OpenBEATs Fine-tuning ==="
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source ~/slurm_env
cd ~/representation-learning

# Set up the virtual environment using uv
uv sync

# Set environment variables
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Run training with the OpenBEATs config (uses GCS bucket directly)
srun uv run representation_learning/run_train.py \
    --config configs/run_configs/pretrained/openbeats_ft.yml

echo "=== Training Complete ==="
echo "End time: $(date)"
