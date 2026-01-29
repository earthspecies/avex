#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-beats"
#SBATCH --cpus-per-gpu=12

cd ~/avex
uv sync
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
srun uv run avex/run_train.py --config configs/run_configs/beats_base_beans.yml