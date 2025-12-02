#!/usr/bin/env bash

#SBATCH --array=0-11
#SBATCH --partition=a100-40
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name="inference_silentcities"
#SBATCH --cpus-per-gpu=12
#SBATCH --output="/home/%u/logs/%A_%a.log"
# cd ..
uv sync
srun uv run scripts/inference_silentcities.py --shard $SLURM_ARRAY_TASK_ID