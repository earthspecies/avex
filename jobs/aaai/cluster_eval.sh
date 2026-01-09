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
# --checkpoint_dir runs/aaai/sl_efficientnet_animalspeak_audioset/2025-07-11_11-42-34/ \
# --device cuda \
uv sync
# srun uv run avex/run_train.py --config configs/run_configs/aaai_train/sl_efficientnet_audioset.yml
# srun uv run avex/run_train.py --config configs/run_configs/aaai_train/sl_efficientnet_animalspeak_audioset.yml
srun uv run python scripts/evaluate_clustering_checkpoints.py \
    --config configs/run_configs/aaai_train/sl_efficientnet_audioset.yml \
    --checkpoint_dir runs/efficientnet_audioset/2025-07-13_07-19-26/ \
    --device cuda \
    --wandb_project avex \
    --wandb_run_name efficientnet_audioset_clustering