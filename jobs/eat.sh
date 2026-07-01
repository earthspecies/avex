#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=2
#SBATCH --ntasks-per-gpu=2
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="eat pretraining"
#SBATCH --cpus-per-gpu=12
cd ~/avex
# ESP-only: uncomment for faster dataset access when you have bucket permissions.
# export ALP_DATA_HOME="gs://esp-ml-datasets/"
uv sync
srun uv run avex train --config configs/run_configs/eat_pretrain_all.yml
