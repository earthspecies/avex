#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="representation learning"
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G

cd ~/avex
uv sync
srun uv run avex train --config configs/run_configs/efficientnet_base.yml
