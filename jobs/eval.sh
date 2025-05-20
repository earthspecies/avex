#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="clip eval 5e5"

cd ~/representation-learning
uv sync
srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/test.yml