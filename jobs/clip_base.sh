#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-clip"
#SBATCH --cpus-per-gpu=10

cd ~/representation-learning
uv sync
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
srun uv run representation_learning/run_train.py --config configs/run_configs/clip_base_beans.yml