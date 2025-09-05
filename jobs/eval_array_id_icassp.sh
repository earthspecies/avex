#!/usr/bin/env bash

#SBATCH --array=1-9%4
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A_%a.log"
#SBATCH --job-name="rl-eval-array-id-icassp"
#SBATCH --cpus-per-gpu=9
#SBATCH --nodelist=slurm-8x-a100-40gb-1

# Map array task ID to config file
declare -A configs=(
    [1]="sl_efficientnet_animalspeak_audioset.yml"
    [2]="ssl_eat_all.yml"
    [3]="bird_aves_bio.yml"
    [4]="beats.yml"
    [5]="beats_naturelm.yml"
    [6]="eat_hf.yml"
    [7]="eat_hf_finetuned.yml"
    [8]="sl_beats_all.yml"
    [9]="sl_eat_all_ssl_all.yml"
)

declare -A configs_ft=(
    [1]="sl_efficientnet_animalspeak_audioset_ft.yml"
    [2]="ssl_eat_all_ft.yml"
    [3]="bird_aves_bio_ft.yml"
    [4]="beats_ft.yml"
    [5]="beats_naturelm_ft.yml"
    [6]="eat_hf_ft.yml"
    [7]="eat_hf_finetuned_ft.yml"
    [8]="sl_beats_all_ft.yml"
    [9]="sl_eat_all_ssl_all_ft.yml"
)

# Get the config file for this array task
config_file=${configs[$SLURM_ARRAY_TASK_ID]}

if [ -z "$config_file" ]; then
    echo "Error: No config file found for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running evaluation for model: $config_file (Task ID: $SLURM_ARRAY_TASK_ID)"

cd ~/code/representation-learning

uv sync

srun uv run repr-learn evaluate --config configs/evaluation_configs/icassp/$config_file --patch dataset_config=configs/data_configs/individual_id_icassp.yml

