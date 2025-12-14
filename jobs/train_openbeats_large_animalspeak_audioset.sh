#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="train-openbeats-large-anispk"

set -euo pipefail
source ~/slurm_env

cd ~/representation-learning
uv sync --group project-dev

CONFIG=${1:-"configs/run_configs/aaai_train/sl_openbeats_large_animalspeak_audioset.yml"}

srun uv run repr-learn train --config "${CONFIG}"

