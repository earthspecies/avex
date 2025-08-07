#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --qos=aaai-2026
#SBATCH --gpus=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eval-birdset"
#SBATCH --cpus-per-gpu=14
uv tool install keyring --with keyrings.google-artifactregistry-auth
# export GOOGLE_APPLICATION_CREDENTIALS=/home/marius_miron_earthspecies_org/.config/gcloud/application_default_credentials.json
# export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

cd ~/code/representation-learning
uv sync
echo $UV_PROJECT_ENVIRONMENT
export PYTHONBUFFERED=1

srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/perch.yml --patch dataset_config=configs/data_configs/benchmark_birdset_gs.yml

# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_model/efficientnet_beans.yml

# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/eat_hf_48khz.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/clap.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/birdnet.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/eat_hf.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/beats.yml
# Alternative single_model configs:
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_model/efficientnet_beans.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/eat_hf_48khz.yml
