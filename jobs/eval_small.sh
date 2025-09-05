#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eval-probe"
#SBATCH --cpus-per-gpu=9
#SBATCH --nodelist=slurm-8x-a100-40gb-1

cd ~/code/representation-learning
uv sync
echo $UV_PROJECT_ENVIRONMENT
# export PYTHONBUFFERED=1
# export LOG_LEVEL=DEBUG
# export PYTHONLOGLEVEL=DEBUG

# srun uv run repr-learn evaluate --config configs/evaluation_configs/icassp/sl_beats_all_ft.yml  --patch dataset_config=configs/data_configs/benchmark_single.yml
# srun uv run repr-learn evaluate --config configs/evaluation_configs/icassp/sl_efficientnet_animalspeak_audioset_ft.yml  --patch dataset_config=configs/data_configs/benchmark_single.yml
# srun uv run repr-learn evaluate --config configs/evaluation_configs/icassp/sl_eat_all_ssl_all.yml --patch dataset_config=configs/data_configs/benchmark_single.yml
srun uv run repr-learn evaluate --config configs/evaluation_configs/icassp/bird_aves_bio.yml  --patch dataset_config=configs/data_configs/benchmark_single.yml
