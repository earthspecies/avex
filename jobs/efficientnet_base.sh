#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-effnet"
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=256GB

cd ~/representation-learning
uv sync
# srun uv run representation_learning/run_train.py --config configs/run_configs/aaai_train/sl_efficientnet_animalspeak.yml
srun uv run representation_learning/run_train.py --config configs/run_configs/aaai_train/sl_efficientnet_animalspeak_audioset.yml