#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="samplerate"
#SBATCH --cpus-per-task=15

cd ~/representation-learning
uv sync
srun uv run scripts/fix_paths.py