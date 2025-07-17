#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-clip-single"

cd ~/representation-learning
uv sync
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Force single GPU mode - no distributed training
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export RANK=0

srun uv run representation_learning/run_train.py --config configs/run_configs/clip_caption_16khz.yml