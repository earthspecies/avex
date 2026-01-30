#!/usr/bin/env bash

#SBATCH --array=4-7
#SBATCH --partition=a100-40
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name="inference_pseudovox"
#SBATCH --cpus-per-gpu=12
#SBATCH --output="/home/%u/logs/%A_%a.log"
# cd ..
uv sync --extra dev
srun uv run scripts/inference_pseudovox_v1.py --shard $SLURM_ARRAY_TASK_ID