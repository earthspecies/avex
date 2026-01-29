#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="sweep example"

cd ~/avex
uv sync
srun uv run esp-sweep agent dazzling-wildebeest
