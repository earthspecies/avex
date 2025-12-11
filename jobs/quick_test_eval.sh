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
#   MODEL_SIZE=large ./jobs/quick_test_eval.sh   # Test with large model
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
MODEL_ID="${MODEL_ID:-openbeats-base-i3}"
MODEL_SIZE="${MODEL_SIZE:-base}"
DEVICE="cuda"

if [ "$USE_CPU" = true ]; then
    DEVICE="cpu"
    echo "Running on CPU (this will be slower)..."
fi

echo "============================================================"
echo "QUICK TEST: Evaluation Pipeline Validation"
echo "============================================================"
echo "Model ID:    $MODEL_ID"
echo "Model Size:  $MODEL_SIZE"
echo "Device:      $DEVICE"
echo "============================================================"

# Sync environment
echo "Syncing environment..."
uv sync# Run quick test evaluation
echo ""
echo "Running quick evaluation (1 epoch, minimal data)..."
echo ""

uv run python -m representation_learning.run_evaluate \
    --config configs/run_configs/pretrained/openbeats_quick_test.yml \
    --model_spec.model_id "$MODEL_ID" \
    --model_spec.model_size "$MODEL_SIZE" \
    --model_spec.device "$DEVICE" \
    --run_name "quick_test_${MODEL_ID}" \
    2>&1 | tee /tmp/quick_test_eval.log

EXIT_CODE=${PIPESTATUS[0]}

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
    echo "Log saved to: /tmp/quick_test_eval.log"
    echo ""
    echo "Common issues:"
    echo "  - Model loading errors: Check MODEL_ID and MODEL_SIZE"
    echo "  - Data loading errors: Check dataset paths and BEANS access"
    echo "  - CUDA errors: Try --cpu flag"
fi

exit $EXIT_CODE
