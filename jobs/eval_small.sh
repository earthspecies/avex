#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eval"
#SBATCH --cpus-per-gpu=6

cd ~/representation-learning
uv sync
srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/birdmae.yml --patch dataset_config=configs/data_configs/benchmark_single.yml
