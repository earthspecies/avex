#!/usr/bin/env bash

#SBATCH --partition=h100-80
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-posttrain-beats"
#SBATCH --cpus-per-gpu=26

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_NVML_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1

cd ~/avex

export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export WANDB_DIR=/scratch-representation-learning/wandb_tmp
export WANDB_CACHE_DIR=/scratch-representation-learning/wandb_cache

uv sync
srun uv run avex/run_train.py --config configs/run_configs/aaai_train/sl_eat_animalspeak_ssl_all_h100.yml