#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-eval-apn-beans"
#SBATCH --cpus-per-gpu=12

cd ~/code/representation_learning

uv tool install keyring --with keyrings.google-artifactregistry-auth
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/representation-learning
export GOOGLE_APPLICATION_CREDENTIALS=/home/marius_miron_earthspecies_org/.config/gcloud/application_default_credentials.json
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export ESP_DATA_HOME='gs://esp-ml-datasets'

srun bash -lc "
  set -euo pipefail
  cd ~/code/representation_learning
  export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/representation-learning
  export GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS
  export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
  export ESP_DATA_HOME='gs://esp-ml-datasets'
  source jobs/setup_slurm_cuda124_venv.sh
  uv run --no-sync avex evaluate \
    --config configs/evaluation_configs/single_models_beans/audioprotopnet.yml \
    --patch dataset_config=configs/data_configs/benchmark_beans_slurm.yml
"
