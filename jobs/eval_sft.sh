#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eval"
#SBATCH --cpus-per-gpu=6


cd ~/avex
# ESP-only: uncomment for faster dataset access when you have bucket permissions.
# export ALP_DATA_HOME="gs://esp-ml-datasets/"
uv sync
srun uv run avex evaluate --config configs/evaluation_configs/test_sft.yml
