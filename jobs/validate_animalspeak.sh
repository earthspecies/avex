#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=2
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-validation"
#SBATCH --cpus-per-gpu=12

export GOOGLE_APPLICATION_CREDENTIALS=/home/david_earthspecies_org/.config/gcloud/application_default_credentials.json
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

cd ~/representation-learning
uv tool install keyring --with keyrings.google-artifactregistry-auth
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/
uv sync
srun uv run scripts/validate_animalspeak.py