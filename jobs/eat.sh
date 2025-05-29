#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="eat pretraining"
#SBATCH --cpus-per-gpu=10

cd ~/representation-learning
uv sync
srun uv run representation_learning/run_train.py --config configs/run_configs/eat_pretrain.yml