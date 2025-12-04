#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/openbeats_beans_%j.log"
#SBATCH --job-name="openbeats-beans-eval"
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# OpenBEATs Evaluation on BEANS Benchmark
# Evaluates FINETUNED OpenBEATs (sl_openbeats_animalspeak) with frozen encoder + linear probe
# on bioacoustic classification and detection tasks
#
# WORKFLOW:
# 1. First run openbeats_ft_single_gpu.sh to finetune OpenBEATs on AnimalSpeak
# 2. Then run this script to evaluate the finetuned model on BEANS
#
# To evaluate pretrained (no finetuning), use:
#   --config configs/evaluation_configs/single_models_beans/openbeats.yml

echo "=== OpenBEATs BEANS Evaluation ==="
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo "GPU info: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Setup environment
cd ~/representation-learning
uv tool install keyring --with keyrings.google-artifactregistry-auth 2>/dev/null || true
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/
export GOOGLE_APPLICATION_CREDENTIALS=/home/$USER/.config/gcloud/application_default_credentials.json
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Sync dependencies
uv sync

# Run evaluation on BEANS benchmark
# Uses frozen FINETUNED OpenBEATs encoder with linear probe
srun uv run repr-learn evaluate \
    --config configs/evaluation_configs/single_models_beans/sl_openbeats_animalspeak.yml \
    --patch dataset_config=configs/data_configs/benchmark_beans.yml

echo "=== Evaluation Complete ==="
echo "End time: $(date)"
