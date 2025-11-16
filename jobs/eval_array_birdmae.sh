#!/usr/bin/env bash

#SBATCH --array=3-4
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A_%a.log"
#SBATCH --job-name="rl-eval-birdmae-array"
#SBATCH --cpus-per-gpu=12

# Map array task ID to config file
declare -A dataset_configs=(
    [1]="benchmark_birdset_gs.yml"
    [2]="beans_detection_gs.yml"
    [3]="benchmark_id_repertoire_with_clustering.yml"
    [4]="beans_classification_gs.yml"
)

# Get the config file for this array task
dataset_config_file=${dataset_configs[$SLURM_ARRAY_TASK_ID]}

if [ -z "$dataset_config_file" ]; then
    echo "Error: No dataset config file found for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running evaluation for dataset: $dataset_config_file (Task ID: $SLURM_ARRAY_TASK_ID)"

cd ~/representation-learning-iclr
uv sync
srun uv run repr-learn evaluate --config configs/evaluation_configs/single_models_beans/birdmae.yml --patch dataset_config=configs/data_configs/$dataset_config_file
