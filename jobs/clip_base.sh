#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus-per-node=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=12
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-clip"

cd ~/representation-learning
uv sync
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# NCCL debugging and configuration for better GPU detection
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

# Ensure GPU visibility is properly set
export CUDA_DEVICE_ORDER=PCI_BUS_ID

srun uv run representation_learning/run_train.py --config configs/run_configs/clip_base_beans.yml