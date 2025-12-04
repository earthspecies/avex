#!/usr/bin/env bash

# Slurm job script for OpenBEATs supervised fine-tuning on a single GPU
# Usage: sbatch jobs/openbeats_ft_single_gpu.sh
#
# This script copies AnimalSpeak data from GCS to local scratch for faster training

#SBATCH --partition=a100-40      # GPU partition - need larger GPU for OpenBEATs-Large
#SBATCH --gpus=1                  # Single GPU
#SBATCH --cpus-per-gpu=8          # CPU cores per GPU
#SBATCH --mem=64G                 # Memory (increased for large model)
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-openbeats-ft"
#SBATCH --time=24:00:00          # Max job duration (increased for large dataset)

echo "=== OpenBEATs Fine-tuning ==="
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source ~/.slurm_env
cd ~/representation-learning

# Set up the virtual environment using uv
uv sync

# Set environment variables
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# =============================================================================
# STEP 1: Copy AnimalSpeak data from GCS to local scratch for fast I/O
# =============================================================================
SCRATCH_DATA_DIR="/scratch/$USER/animalspeak_16k"
GCS_DATA_PATH="gs://foundation-model-data/audio_16k/animalspeak2"

# Check if data already exists on scratch (from a previous job)
if [ -d "$SCRATCH_DATA_DIR/16khz" ] && [ "$(ls -A $SCRATCH_DATA_DIR/16khz 2>/dev/null)" ]; then
    echo "Data already exists at $SCRATCH_DATA_DIR, skipping download..."
    echo "Contents: $(ls $SCRATCH_DATA_DIR | head -5)..."
else
    echo "Copying AnimalSpeak data from GCS to local scratch..."
    echo "Source: $GCS_DATA_PATH"
    echo "Destination: $SCRATCH_DATA_DIR"
    
    mkdir -p "$SCRATCH_DATA_DIR"
    
    # Use gcloud storage cp for parallel downloads (faster than gsutil)
    # -r for recursive
    gcloud storage cp -r "$GCS_DATA_PATH/*" "$SCRATCH_DATA_DIR/" 
    
    echo "Data copy complete. Size: $(du -sh $SCRATCH_DATA_DIR | cut -f1)"
fi

# =============================================================================
# STEP 2: Create a temporary data config pointing to scratch
# =============================================================================
SCRATCH_DATA_CONFIG="/scratch/$USER/data_config_scratch_$SLURM_JOB_ID.yml"

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

echo "Created scratch data config at $SCRATCH_DATA_CONFIG"

# =============================================================================
# STEP 3: Run training with scratch data config
# =============================================================================
echo "Starting training at $(date)..."

# Run training with the OpenBEATs config, using scratch data config
srun uv run representation_learning/run_train.py \
    --config configs/run_configs/pretrained/openbeats_ft.yml \
    --patch dataset_config="$SCRATCH_DATA_CONFIG"

echo "=== Training Complete ==="
echo "End time: $(date)"

# Cleanup temp config (optional - scratch is ephemeral anyway)
rm -f "$SCRATCH_DATA_CONFIG"
