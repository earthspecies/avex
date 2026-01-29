#!/usr/bin/env bash

# --------------------------------------------------------------------------- #
#  Slurm array job – BEANS benchmark (EfficientNet / Multi-label / BEATs)
# --------------------------------------------------------------------------- #
#  Index → config mapping:
#   0 → EfficientNet-B0 baseline                (efficientnet_base_beans.yml)
#   1 → EfficientNet-B0 multi-label experiment  (efficientnet_multilabel.yml)
#   2 → BEATs backbone baseline                 (beats_base_beans.yml)
# --------------------------------------------------------------------------- #

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A_%a.log"
#SBATCH --job-name="beans_train_array"
#SBATCH --array=0-2

# Navigate to repository root on compute node
cd ~/avex

# Ensure dependencies are in sync on the worker
uv sync

# --------------------------- config selector ------------------------------- #
CONFIG=""
case "$SLURM_ARRAY_TASK_ID" in
    0)
        CONFIG="configs/run_configs/efficientnet_base_beans.yml"
        ;;
    1)
        CONFIG="configs/run_configs/efficientnet_multilabel.yml"
        ;;
    2)
        CONFIG="configs/run_configs/beats_base_beans.yml"
        ;;
    *)
        echo "[Error] Unsupported SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
        exit 2
        ;;
esac

# --------------------------- launch training -------------------------------- #

echo "Launching training with config: $CONFIG"

srun uv run avex/run_train.py --config "$CONFIG"