#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-efffnet"
#SBATCH --cpus-per-gpu=12


cd ~/representation-learning
# export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/
# export UV_CACHE_DIR=/scratch/$USER/uv_cache/

uv sync
srun uv run repr-learn train --config  configs/run_configs/aaai_train/sl_efficientnet_animalspeak_audioset.yml
