#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=2
#SBATCH --ntasks=2
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="representation learning clip"

cd ~/representation-learning
uv sync
srun uv run representation_learning/run_train.py --config configs/run_configs/clip_base_beans.yml