#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="aves eval"

cd ~/code/avex
uv sync
export ESP_DATA_HOME="gs://esp-ml-datasets"
srun uv run avex evaluate --config configs/evaluation_configs/aves_bio.yml
