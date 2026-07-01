#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="sweep example"

cd ~/avex
# ESP-only: uncomment for faster dataset access when you have bucket permissions.
# export ALP_DATA_HOME="gs://esp-ml-datasets/"
uv sync
srun uv run esp-sweep agent dazzling-wildebeest
