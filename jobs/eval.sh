#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eval"
#SBATCH --cpus-per-gpu=10

cd ~/representation-learning
uv tool install keyring --with keyrings.google-artifactregistry-auth
uv sync
srun uv run repr-learn evaluate --config configs/evaluation_configs/single_model/efficientnet_beans.yml
# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/eat_hf_48khz.yml
