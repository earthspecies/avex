#!/usr/bin/env bash

#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-effnet-test"
#SBATCH --cpus-per-gpu=12

cd ~/representation-learning
uv sync
srun uv run representation_learning/run_train.py --config configs/run_configs/efficientnet_base_test.yml