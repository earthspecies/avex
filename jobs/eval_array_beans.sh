#!/usr/bin/env bash

#SBATCH --array=1-9%3
#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A_%a.log"
#SBATCH --job-name="rl-eval-array"
#SBATCH --cpus-per-gpu=10

# Map array task ID to config file
declare -A configs=(
    # [1]="efficientnet_beans.yml"
    # [2]="eat_hf.yml"
    # [3]="atst_frame.yml"
    # [4]="clap.yml"
    [5]="beats.yml"
    # [6]="bird_aves_bio.yml"
    # [7]="beats_naturelm.yml"
    # [8]="beats_finetuned.yml"
    # [9]="perch.yml"
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
uv sync
srun uv run representation_learning/run_evaluate.py --config configs/evaluation_configs/single_models_beans/$config_file
