#!/usr/bin/env bash

# Slurm job array for OpenBEATs experiments
# Downloads data once per node, then runs multiple configs
#
# Usage: sbatch jobs/openbeats_ft_array.sh
#
# This uses job arrays to run multiple experiments efficiently:
# - First task (ID 1) downloads data and runs experiment 1
# - Subsequent tasks on same node reuse cached data
#
# Modify the configs array below to add/change experiments

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --output="/home/%u/logs/openbeats_%A_%a.log"
#SBATCH --job-name="openbeats-array"
#SBATCH --time=24:00:00
#SBATCH --array=1-2%1            # Run 2 experiments, 1 at a time (sequential on same node)

# =============================================================================
# Define experiments - modify this array to add more configs
# =============================================================================
declare -A experiments=(
    [1]="openbeats-large-i2"     # Iteration 2
    [2]="openbeats-large-i3"     # Iteration 3
)

# Get current experiment
EXPERIMENT=${experiments[$SLURM_ARRAY_TASK_ID]}

if [ -z "$EXPERIMENT" ]; then
    echo "Error: No experiment defined for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== OpenBEATs Fine-tuning Array ==="
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Experiment: $EXPERIMENT"

source ~/.slurm_env
cd ~/representation-learning

# Set up the virtual environment using uv
uv sync

export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# =============================================================================
# STEP 1: Copy AnimalSpeak data from GCS to local scratch (shared across array tasks)
# =============================================================================
SCRATCH_DATA_DIR="/scratch/$USER/animalspeak_16k"
GCS_DATA_PATH="gs://foundation-model-data/audio_16k/animalspeak2"
DOWNLOAD_LOCK="/scratch/$USER/.animalspeak_download.lock"

# Use a lock file to prevent multiple array tasks from downloading simultaneously
if [ -d "$SCRATCH_DATA_DIR/16khz" ] && [ "$(ls -A $SCRATCH_DATA_DIR/16khz 2>/dev/null)" ]; then
    echo "Data already exists at $SCRATCH_DATA_DIR, skipping download..."
else
    # Try to acquire lock (only first task should download)
    if mkdir "$DOWNLOAD_LOCK" 2>/dev/null; then
        echo "Acquired download lock, copying data from GCS..."
        echo "Source: $GCS_DATA_PATH"
        echo "Destination: $SCRATCH_DATA_DIR"
        
        mkdir -p "$SCRATCH_DATA_DIR"
        gcloud storage cp -r "$GCS_DATA_PATH/*" "$SCRATCH_DATA_DIR/"
        
        echo "Data copy complete. Size: $(du -sh $SCRATCH_DATA_DIR | cut -f1)"
        rmdir "$DOWNLOAD_LOCK"
    else
        echo "Another task is downloading, waiting..."
        while [ -d "$DOWNLOAD_LOCK" ]; do
            sleep 30
        done
        echo "Download complete, proceeding..."
    fi
fi

# =============================================================================
# STEP 2: Create experiment-specific config
# =============================================================================
SCRATCH_DATA_CONFIG="/scratch/$USER/data_config_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yml"

cat > "$SCRATCH_DATA_CONFIG" << EOF
train_datasets:
  - dataset_name: animalspeak
    dataset_version: 0.0
    split: train
    balance: false
    balance_attribute: canonical_name
    custom_balancing: false
    balancing_method: upsample
    subset_percentage: 1.0
    audio_max_length_seconds: 10
    audio_path_col: path
    data_root: ${SCRATCH_DATA_DIR}/
    transformations:
      - type: label_from_feature
        feature: canonical_name
        override: true

val_datasets:
  - dataset_name: animalspeak
    dataset_version: 0.0
    split: validation
    balance: false
    balance_attribute: canonical_name
    custom_balancing: false
    balancing_method: upsample
    subset_percentage: 1.0
    audio_max_length_seconds: 10
    audio_path_col: path
    data_root: ${SCRATCH_DATA_DIR}/
    transformations:
      - type: label_from_feature
        feature: canonical_name
        override: true

concatenate_train: true
concatenate_val: true
concatenate_method: soft
EOF

# =============================================================================
# STEP 3: Run training with experiment-specific model
# =============================================================================
echo "Starting training at $(date)..."
echo "Model: $EXPERIMENT"

# Determine model size from experiment name
if [[ "$EXPERIMENT" == *"large"* ]]; then
    MODEL_SIZE="large"
elif [[ "$EXPERIMENT" == *"base"* ]]; then
    MODEL_SIZE="base"
else
    MODEL_SIZE="large"
fi

# Run training, patching model_id and output directory
srun uv run representation_learning/run_train.py \
    --config configs/run_configs/pretrained/openbeats_ft.yml \
    --patch dataset_config="$SCRATCH_DATA_CONFIG" \
    --patch model_spec.model_id="$EXPERIMENT" \
    --patch model_spec.model_size="$MODEL_SIZE" \
    --patch run_name="openbeats_ft_${EXPERIMENT}" \
    --patch output_dir="./runs/openbeats_ft_${EXPERIMENT}"

echo "=== Training Complete ==="
echo "End time: $(date)"

rm -f "$SCRATCH_DATA_CONFIG"
