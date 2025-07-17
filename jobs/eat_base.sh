#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eat"
#SBATCH --cpus-per-gpu=12

cd ~/representation-learning

uv tool install keyring --with keyrings.google-artifactregistry-auth
# export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/
# export UV_CACHE_DIR=/scratch/$USER/uv_cache/
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1


uv sync
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
srun uv run representation_learning/run_train.py --config configs/run_configs/aaai_train/sl_eat_animalspeak_audioset.yml