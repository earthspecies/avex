#!/usr/bin/env bash
# ==============================================================================
# Quick Test Evaluation Script
# ==============================================================================
# Fast structural validation of the evaluation pipeline.
# Runs 1 epoch with minimal data to verify everything works before full runs.
#
# Usage:
#   ./jobs/quick_test_eval.sh                    # Run directly (interactive)
#   sbatch jobs/quick_test_eval.sh               # Submit to SLURM
#   ./jobs/quick_test_eval.sh --cpu              # Run on CPU (slower but no GPU needed)
#
# Expected runtime: 1-3 minutes on GPU, 5-10 minutes on CPU
# ==============================================================================

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output="/home/%u/logs/quick_test_%j.log"
#SBATCH --job-name="quick-test"
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G

set -e

# Source environment (for SLURM jobs)
if [ -f ~/slurm_env ]; then
    source ~/slurm_env
fi

# Navigate to repo root (works for both interactive and SLURM)
cd ~/representation-learning

# Parse arguments
USE_CPU=false
for arg in "$@"; do
    case $arg in
        --cpu)
            USE_CPU=true
            shift
            ;;
    esac
done

# Configuration
DEVICE="cuda"
if [ "$USE_CPU" = true ]; then
    DEVICE="cpu"
fi

echo "============================================================"
echo "QUICK TEST: Evaluation Pipeline Validation"
echo "============================================================"
echo "Device:      $DEVICE"
echo "Config:      configs/evaluation_configs/quick_test.yml"
echo "============================================================"

# Sync environment
echo "Syncing environment..."
uv sync

# Run quick test evaluation using proper evaluation config
echo ""
echo "Running quick evaluation (1 epoch, minimal data)..."
echo ""

uv run python -m representation_learning.cli evaluate \
    --config configs/evaluation_configs/quick_test.yml \
    --patch "device=$DEVICE"

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ QUICK TEST PASSED - Pipeline is structurally correct"
    echo "============================================================"
    echo "You can now run full evaluations with confidence."
    echo ""
    echo "Next steps:"
    echo "  ./jobs/openbeats_full_eval.sh beans"
    echo "  ./jobs/openbeats_full_eval.sh all"
else
    echo "❌ QUICK TEST FAILED - Check logs above for errors"
    echo "============================================================"
    echo ""
    echo "Common issues:"
    echo "  - Model loading errors: Check HuggingFace access"
    echo "  - Data loading errors: Check dataset paths and BEANS access"
    echo "  - CUDA errors: Try --cpu flag"
fi

exit $EXIT_CODE
