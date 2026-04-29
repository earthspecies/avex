#!/usr/bin/env bash

#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=26
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="clap-chain-h100"
#SBATCH --qos=naturelm

cd ~/avex
uv sync --group project-dev --group gpu

export PYTORCH_DISTRIBUTED_BACKEND=nccl
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

srun uv run avex train \
    --config configs/run_configs/clap/clap_chain_synthetic_xc_inat.yml \
    -p training_params.batch_size=680 \
    -p num_workers=24
