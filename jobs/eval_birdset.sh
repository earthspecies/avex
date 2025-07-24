#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eval-birdset"
#SBATCH --cpus-per-gpu=12


cd ~/representation-learning
uv sync

srun uv run repr-learn evaluate --config configs/evaluation_configs/efficientnet_bio_birdset.yml

# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_model/efficientnet_beans.yml

# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/eat_hf_48khz.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/clap.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/birdnet.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/eat_hf.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/beats.yml
# Alternative single_model configs:
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_model/efficientnet_beans.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/eat_hf_48khz.yml
