#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-clip-robust"

cd ~/representation-learning
uv sync
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Comprehensive NCCL configuration for better reliability
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=0

# CUDA configuration
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# Alternative: try gloo backend instead of NCCL
export PYTORCH_DISTRIBUTED_BACKEND=nccl

# GPU binding with explicit task mapping
srun --cpu-bind=verbose --gpu-bind=closest uv run avex/run_train.py --config configs/run_configs/clip_base_beans.yml