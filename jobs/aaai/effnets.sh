#!/usr/bin/env bash

#SBATCH --array=1-3%3
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A_%a.log"
#SBATCH --job-name="rl-effnet-array"
#SBATCH --cpus-per-gpu=12

# Map array task ID to config file
declare -A configs=(
    [1]="sl_efficientnet_animalspeak_audioset.yml"
    [2]="sl_efficientnet_audioset.yml"
    [3]="sl_efficientnet_animalspeak.yml"
)



# Get the config file for this array task
config_file=${configs[$SLURM_ARRAY_TASK_ID]}

if [ -z "$config_file" ]; then
    echo "Error: No config file found for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running EfficientNet training for config: $config_file (Task ID: $SLURM_ARRAY_TASK_ID)"

cd ~/representation-learning

# Environment setup
export GOOGLE_APPLICATION_CREDENTIALS=/home/david_earthspecies_org/.config/gcloud/application_default_credentials.json
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export WANDB_DIR=/scratch-representation-learning/wandb_tmp
export WANDB_CACHE_DIR=/scratch-representation-learning/wandb_cache

# Install dependencies
uv tool install keyring --with keyrings.google-artifactregistry-auth
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/
uv sync

# Run training
srun uv run rep-learn train --config configs/run_configs/aaai_train/$config_file
