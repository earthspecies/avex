#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="eval-large-i2"
#SBATCH --cpus-per-gpu=9

set -euo pipefail
source ~/slurm_env

cd ~/representation-learning
uv sync --group project-dev

# Default values (can be overridden by args)
CHECKPOINT=${1:-""}  # Not used for HF OpenBEATs; kept for interface parity
DATASET_CONFIG=${2:-"configs/data_configs/benchmark.yml"}
EVAL_CONFIG=${3:-"configs/evaluation_configs/icassp/openbeats_large_i2.yml"}
SAVE_DIR=${4:-"evaluation_results/icassp/openbeats_large_i2"}

srun uv run repr-learn evaluate --config ${EVAL_CONFIG} || exit 1

