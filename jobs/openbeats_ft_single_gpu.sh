#!/usr/bin/env bash

# Slurm job script for OpenBEATs supervised fine-tuning on a single GPU
# Usage: sbatch jobs/openbeats_ft_single_gpu.sh

#SBATCH --partition=t4           # GPU partition (t4, a100-40, a100-80, h100-80)
#SBATCH --gpus=1                  # Single GPU
#SBATCH --cpus-per-gpu=8          # CPU cores per GPU
#SBATCH --mem=32G                 # Memory
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-openbeats-ft"
#SBATCH --time=12:00:00          # Max job duration

source ~/.slurm_env
cd ~/representation-learning

# Set up the virtual environment using uv
uv sync

# Set environment variables
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Run training with the OpenBEATs config
srun uv run representation_learning/run_train.py \
    --config configs/run_configs/pretrained/openbeats_ft.yml
