#!/usr/bin/env bash

#SBATCH --array=1-28%6
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A_%a.log"
#SBATCH --job-name="rl-eval-array"
#SBATCH --cpus-per-gpu=12

# Map array task ID to config file
declare -A configs=(
    # [2]="sl_efficientnet_animalspeak.yml"
    # [3]="sl_efficientnet_animalspeak_audioset.yml"
    # [4]="ssl_eat_all.yml"
    # [5]="ssl_eat_animalspeak.yml"
    # [6]="ssl_eat_audioset.yml"
    # [7]="bird_aves_bio.yml"
    # [9]="beats.yml"
    # [11]="beats_naturelm.yml" 
    # [12]="beats_finetuned.yml"
    # [13]="perch.yml"
    # [14]="sl_efficientnet_audioset.yml"
    # [15]="eat_hf.yml"
    # [16]="eat_hf_finetuned.yml"
    # [1]="birdnet.yml"
    # [18]="sl_beats_animalspeak.yml"
    # [19]="sl_beats_all.yml"
    # [20]="sl_eat_all_ssl_all.yml"
    # [21]="sl_eat_animalspeak_ssl_all.yml"
    # [22]="sl_efficientnet_animalspeak_soundscape.yml"
    # [23]="sl_efficientnet_animalspeak_wabad.yml"
    [24]="sl_efficientnet_animalspeak_nowhales.yml"
    # [25]="sl_efficientnet_animalspeak_nobirds.yml"
    [26]="sl_efficientnet_animalspeak_birds.yml"
    # [25]="consolidated_beans_models_part1.yml"
    # [26]="consolidated_beans_models_part2.yml"
    # [27]="consolidated_beans_models_part3.yml"
)

# Get the config file for this array task
config_file=${configs[$SLURM_ARRAY_TASK_ID]}

if [ -z "$config_file" ]; then
    echo "Error: No config file found for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running evaluation for model: $config_file (Task ID: $SLURM_ARRAY_TASK_ID)"

cd ~/rep5
uv tool install keyring --with keyrings.google-artifactregistry-auth
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/
export GOOGLE_APPLICATION_CREDENTIALS=/home/david_earthspecies_org/.config/gcloud/application_default_credentials.json
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
uv sync
# Check if it's a consolidated config or single model config
srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/$config_file --patch dataset_config=configs/data_configs/benchmark_single.yml
