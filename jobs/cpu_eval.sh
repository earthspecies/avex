#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eval"
#SBATCH --cpus-per-task=30

cd ~/avex
uv sync
srun uv run avex/run_evaluate.py --config configs/evaluation_configs/perch_cpu.yml