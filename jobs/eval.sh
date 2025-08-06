#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eval"
#SBATCH --cpus-per-gpu=12

export GOOGLE_APPLICATION_CREDENTIALS=/home/david_earthspecies_org/.config/gcloud/application_default_credentials.json
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export CLOUDPATHLIB_CACHE_DIR="/tmp/cloudpathlib_$(date +%s)_$$"  # Use unique cache dir
export BEANS_DEBUG=0

cd ~/rep5

uv tool install keyring --with keyrings.google-artifactregistry-auth
uv sync

# Clear cloudpathlib cache for BirdSet files to ensure fresh downloads
uv run python clear_cloudpathlib_cache.py


# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/beats_finetuned.yml --patch dataset_config=configs/data_configs/finch.yml


# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/sl_efficientnet_animalspeak_wabad.yml --patch dataset_config=configs/data_configs/benchmark_probe.yml

# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/eat_hf_48khz.yml
# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/sl_efficientnet_animalspeak.yml --patch dataset_config=configs/data_configs/benchmark_birdset.yml
# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/birdmae.yml --patch dataset_config=configs/data_configs/beans.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/clap.yml
# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/birdnet.yml --patch dataset_config=configs/data_configs/benchmark_birdset_gs_1.yml
# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/surfperch.yml --patch dataset_config=configs/data_configs/individual_id_h100.yml
# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/sl_beats_animalspeak.yml --patch dataset_config=configs/data_configs/individual_id_h100.yml

# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/birdnet.yml --patch dataset_config=configs/data_configs/individual_id_h100.yml
# srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/sl_eat_animalspeak_ssl_all.yml --patch dataset_config=configs/data_configs/benchmark_single.yml
srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/bird_aves_bio.yml --patch dataset_config=configs/data_configs/individual_id_h100.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/eat_hf.yml
# Alternative single_model configs:
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_model/efficientnet_beans.yml
# srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/eat_hf_48khz.yml
# 