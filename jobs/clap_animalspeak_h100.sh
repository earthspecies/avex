#!/usr/bin/env bash

#SBATCH --partition=h100-80
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="clap-as-16k-h100"
#SBATCH --qos=naturelm

uv sync --group project-dev --group gpu

export PYTORCH_DISTRIBUTED_BACKEND=nccl
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

srun uv run avex train \
    --config configs/run_configs/clap/clap_animalspeak.yml \
    -p training_params.batch_size=680 \
    -p num_workers=24
