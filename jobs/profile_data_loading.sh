#!/usr/bin/env bash

# Profile data loading to identify training bottlenecks
# Usage: sbatch jobs/profile_data_loading.sh

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --output="/home/%u/logs/profile_%j.log"
#SBATCH --job-name="profile-data"
#SBATCH --time=0:30:00

echo "=== Data Loading Profiler ==="
echo "Start time: $(date)"
echo "Running on: $(hostname)"

source ~/slurm_env
cd ~/representation-learning

uv sync
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Run the profiler
uv run python scripts/profile_data_loading.py \
    --config configs/run_configs/pretrained/openbeats_ft.yml \
    --num-samples 20 \
    --num-batches 20

echo "=== Profiling Complete ==="
echo "End time: $(date)"
