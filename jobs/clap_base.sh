#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-clap"
#SBATCH --cpus-per-gpu=12

cd ~/representation-learning
# uv tool install keyring --with keyrings.google-artifactregistry-auth
uv sync

# export CLOUDPATHLIB_LOCAL_CACHE_DIR="/scratch/$USER/shared_cache"

# NCCL debugging and configuration for better GPU detection
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_NVML_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
export DEBUG_CLIP_GATHER=1

export PYTORCH_DISTRIBUTED_BACKEND=nccl
export DATASET_DEBUG_SAVE=true

#keyring

export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
srun uv run representation_learning/run_train.py --config configs/run_configs/aaai_train/clap_efficientnet_captions.yml