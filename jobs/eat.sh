#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="eat pretraining"
#SBATCH --cpus-per-gpu=12
# uv tool install keyring --with keyrings.google-artifactregistry-auth --with keyrings.alt --force
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

cd ~/representation-learning
uv sync
srun uv run representation_learning/run_train.py --config configs/run_configs/eat_pretrain_all.yml
