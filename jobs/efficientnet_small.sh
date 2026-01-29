#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-effnet-test"
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=32G  # Reserve 32GB of RAM

cd ~/avex
uv sync
srun uv run avex train --config configs/run_configs/efficientnet_base_test.yml
