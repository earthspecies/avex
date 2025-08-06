#!/usr/bin/env bash

<<<<<<< HEAD
#SBATCH --array=1-25%4
#SBATCH --partition=h100-80
=======
#SBATCH --array=1-18%1
#SBATCH --qos=aaai-2026
#SBATCH --partition=a100-40
>>>>>>> main
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A_%a.log"
#SBATCH --job-name="rl-eval-array"
<<<<<<< HEAD
#SBATCH --cpus-per-gpu=24

# Map array task ID to config file
declare -A configs=(
=======
#SBATCH --cpus-per-gpu=12

# Map array task ID to config file
# Map array task ID to config file
declare -A configs=(
    [1]="sl_efficientnet_audioset.yml"
>>>>>>> main
    [2]="sl_efficientnet_animalspeak.yml"
    [3]="sl_efficientnet_animalspeak_audioset.yml"
    [4]="ssl_eat_all.yml"
    [5]="ssl_eat_animalspeak.yml"
    [6]="ssl_eat_audioset.yml"
    [7]="bird_aves_bio.yml"
<<<<<<< HEAD
    [9]="beats.yml"
    [11]="beats_naturelm.yml"
    [12]="beats_finetuned.yml"
    # [13]="perch.yml"
    [14]="sl_efficientnet_audioset.yml"
    [15]="eat_hf.yml"
    [16]="eat_hf_finetuned.yml"
    # [17]="birdnet.yml"
    [18]="sl_beats_animalspeak.yml"
    [19]="sl_beats_all.yml"
    [20]="sl_eat_all_ssl_all.yml"
    [21]="sl_eat_animalspeak_ssl_all.yml"
=======
    [8]="beats.yml"
    [9]="beats_naturelm.yml"
    [10]="beats_finetuned.yml"
    [11]="eat_hf.yml"
    [12]="eat_hf_finetuned.yml"
    [13]="sl_beats_animalspeak.yml"
    [14]="sl_beats_all.yml"
    [15]="sl_eat_all_ssl_all.yml"
    [16]="sl_eat_animalspeak_ssl_all.yml"
    # [17]="perch.yml"
    # [18]="birdnet.yml"
>>>>>>> main
)

# Get the config file for this array task
config_file=${configs[$SLURM_ARRAY_TASK_ID]}

if [ -z "$config_file" ]; then
    echo "Error: No config file found for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running evaluation for model: $config_file (Task ID: $SLURM_ARRAY_TASK_ID)"

<<<<<<< HEAD
cd ~/rep5
uv tool install keyring --with keyrings.google-artifactregistry-auth
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/
export GOOGLE_APPLICATION_CREDENTIALS=/home/david_earthspecies_org/.config/gcloud/application_default_credentials.json
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

uv sync
=======
cd ~/code/representation_learning
uv tool install keyring --with keyrings.google-artifactregistry-auth
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/
export GOOGLE_APPLICATION_CREDENTIALS=/home/marius_miron_earthspecies_org/.config/gcloud/application_default_credentials.json
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

echo $UV_PROJECT_ENVIRONMENT
uv sync

>>>>>>> main
srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/$config_file --patch dataset_config=configs/data_configs/benchmark_birdset.yml
