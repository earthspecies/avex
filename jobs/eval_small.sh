#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eval-probe"
#SBATCH --cpus-per-gpu=10

cd ~/code/representation-learning
uv sync
echo $UV_PROJECT_ENVIRONMENT
# export PYTHONBUFFERED=1
# export LOG_LEVEL=DEBUG
# export PYTHONLOGLEVEL=DEBUG
srun uv run repr-learn evaluate --config configs/evaluation_configs/icassp/sl_efficientnet_audioset.yml  --patch dataset_config=configs/data_configs/benchmark_single.yml
