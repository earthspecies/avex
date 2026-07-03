#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="samplerate"
#SBATCH --cpus-per-task=15

cd ~/avex
# ESP-only: uncomment for faster dataset access when you have bucket permissions.
# export ALP_DATA_HOME="gs://esp-ml-datasets/"
uv sync
srun uv run scripts/fix_paths.py