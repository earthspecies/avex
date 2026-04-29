#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=12
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="clap-as-16k"

cd ~/avex
uv sync --group project-dev --group gpu

# NCCL configuration for multi-GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_NVML_DISABLE=1

export PYTORCH_DISTRIBUTED_BACKEND=nccl
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

srun uv run avex train \
    --config configs/run_configs/clap/clap_animalspeak.yml \
    -p training_params.batch_size=300 \
    -p num_workers=12
