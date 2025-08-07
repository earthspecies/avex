#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=2
#SBATCH --ntasks-per-gpu=2
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="eat pretraining"
#SBATCH --cpus-per-gpu=12
cd ~/representation-learning
uv sync
srun uv run repr-learn train --config configs/run_configs/eat_pretrain_all.yml
