#!/usr/bin/env bash
#SBATCH --job-name=clap-baseline
#SBATCH --partition=h100-80
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
#SBATCH --cpus-per-task=26
#SBATCH --nodes=1

uv sync --group project-dev --group gpu

export PYTORCH_DISTRIBUTED_BACKEND=nccl
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

srun uv run avex train \
    --config configs/run_configs/clap/clap_chain_animalspeak_audiosetcaps.yml \
    -p training_params.batch_size=680 \
    -p training_params.train_epochs=30 \
    -p num_workers=18
