#!/usr/bin/env bash

#SBATCH --array=1-6%3
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A_%a.log"
#SBATCH --job-name="rl-eval-array"
#SBATCH --cpus-per-gpu=10

# Map array task ID to config file
declare -A configs=(
    [1]="sl_efficientnet_audioset.yml"
    [2]="sl_efficientnet_animalspeak.yml"
    [3]="sl_efficientnet_animalspeak_audioset.yml"
    [4]="ssl_eat_all.yml"
    [5]="ssl_eat_animalspeak.yml"
    [6]="ssl_eat_audioset.yml"
    [7]="atst_frame.yml"
    [8]="clap.yml"
    [9]="beats.yml"
    [10]="bird_aves_bio.yml"
    [11]="beats_naturelm.yml"
    [12]="beats_finetuned.yml"
    [13]="perch.yml"
)

# Get the config file for this array task
config_file=${configs[$SLURM_ARRAY_TASK_ID]}

if [ -z "$config_file" ]; then
    echo "Error: No config file found for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running evaluation for model: $config_file (Task ID: $SLURM_ARRAY_TASK_ID)"

cd ~/representation-learning
uv tool install keyring --with keyrings.google-artifactregistry-auth
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/
export GOOGLE_APPLICATION_CREDENTIALS=/home/marius_miron_earthspecies_org/.config/gcloud/application_default_credentials.json
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
uv sync
srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_birdset/$config_file
