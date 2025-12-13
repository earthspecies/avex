#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eval-beats-all"
#SBATCH --cpus-per-gpu=9

set -euo pipefail
source ~/slurm_env

cd ~/representation-learning
uv sync --group project-dev

# Default values (can be overridden by args)
CHECKPOINT=${1:-"gs://representation-learning/models/sl_beats_all.pt"}
DATASET_CONFIG=${2:-"configs/data_configs/benchmark.yml"}
EVAL_CONFIG=${3:-"configs/evaluation_configs/icassp/sl_beats_all.yml"}
SAVE_DIR=${4:-"evaluation_results/icassp/sl_beats_all"}

srun uv run repr-learn evaluate --config ${EVAL_CONFIG} || exit 1