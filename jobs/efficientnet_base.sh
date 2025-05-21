#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="representation learning effnet"

cd ~/representation-learning
uv sync
srun uv run representation_learning/run_train.py --config configs/run_configs/efficientnet_base_beans.yml