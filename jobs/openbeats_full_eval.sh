#!/usr/bin/env bash
# ==============================================================================
# BEATs / OpenBEATs Evaluation Script
# ==============================================================================
# Unified evaluation script for BEATs and OpenBEATs models on standard benchmarks.
#
# Benchmarks:
#   beans            - BEANS Classification + Detection
#   birdset          - BirdSet Detection
#   individual_id    - Individual ID Classification
#   vocal_repertoire - Vocal Repertoire Classification
#
# Usage:
#   ./jobs/openbeats_full_eval.sh                           # Run all benchmarks (default: openbeats-large-i3)
#   ./jobs/openbeats_full_eval.sh beans                     # Run BEANS only
#   ./jobs/openbeats_full_eval.sh birdset individual_id     # Run multiple benchmarks
#   ./jobs/openbeats_full_eval.sh all                       # Run all benchmarks
#
# Environment Variables:
#   MODEL_ID      - HuggingFace model ID (default: openbeats-large-i3)
#   MODEL_SIZE    - Model size: base or large (default: large)
#   RUN_NAME      - Custom run name for results (default: derived from MODEL_ID)
#   EVAL_CONFIG   - Override evaluation config file
#
# Examples:
#   MODEL_ID=openbeats-base-i3 MODEL_SIZE=base ./jobs/openbeats_full_eval.sh beans
#   RUN_NAME=my_experiment ./jobs/openbeats_full_eval.sh birdset
#
# SLURM:
#   sbatch jobs/openbeats_full_eval.sh [benchmarks...]
# ==============================================================================

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --output="/home/%u/logs/beats_eval_%A_%a.log"
#SBATCH --job-name="beats-eval"
#SBATCH --cpus-per-gpu=12

set -e
source ~/slurm_env

# ==============================================================================
# Configuration
# ==============================================================================

# Model configuration (can be overridden via environment variables)
MODEL_ID="${MODEL_ID:-openbeats-large-i3}"
MODEL_SIZE="${MODEL_SIZE:-large}"
RUN_NAME="${RUN_NAME:-${MODEL_ID}}"

# Evaluation config
EVAL_CONFIG="${EVAL_CONFIG:-configs/evaluation_configs/openbeats_full_eval.yml}"

# Benchmark data configs
declare -A DATA_CONFIGS=(
    ["beans"]="configs/data_configs/beans.yml"
    ["birdset"]="configs/data_configs/benchmark_birdset.yml"
    ["individual_id"]="configs/data_configs/individual_id_icassp.yml"
    ["vocal_repertoire"]="configs/data_configs/benchmark_id_repertoire_with_clustering.yml"
)

ALL_BENCHMARKS="beans birdset individual_id vocal_repertoire"

# ==============================================================================
# Setup
# ==============================================================================

# Navigate to repo root
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
cd ~/representation-learning 2>/dev/null || cd "$(git rev-parse --show-toplevel)"

# Environment setup
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export BEANS_DEBUG=0

uv sync

# ==============================================================================
# Functions
# ==============================================================================

show_help() {
    echo "Usage: $0 [benchmark1] [benchmark2] ..."
    echo ""
    echo "Benchmarks:"
    echo "  all              Run all benchmarks"
    echo "  beans            BEANS Classification + Detection"
    echo "  birdset          BirdSet Detection"
    echo "  individual_id    Individual ID Classification"
    echo "  vocal_repertoire Vocal Repertoire Classification"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_ID      HuggingFace model ID (default: openbeats-large-i3)"
    echo "  MODEL_SIZE    Model size: base or large (default: large)"
    echo "  RUN_NAME      Custom run name (default: derived from MODEL_ID)"
    echo "  EVAL_CONFIG   Override evaluation config file"
    echo ""
    echo "Examples:"
    echo "  $0 beans birdset"
    echo "  MODEL_ID=openbeats-base-i3 MODEL_SIZE=base $0 all"
}

run_eval() {
    local bench=$1
    local data_cfg=${DATA_CONFIGS[$bench]}
    
    if [ -z "$data_cfg" ]; then
        echo "ERROR: Unknown benchmark '$bench'"
        show_help
        exit 1
    fi
    
    local save_dir="evaluation_results/${RUN_NAME}_${bench}"
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Benchmark: $bench"
    echo "  Model:     $MODEL_ID ($MODEL_SIZE)"
    echo "  Save dir:  $save_dir"
    echo "════════════════════════════════════════════════════════════════"
    
    local cmd="uv run repr-learn evaluate \
        --config $EVAL_CONFIG \
        --patch dataset_config=$data_cfg \
        --patch save_dir=$save_dir"
    
    # Add model overrides if non-default
    if [ "$MODEL_ID" != "openbeats-large-i3" ]; then
        cmd="$cmd --patch experiments.0.run_config.model_spec.model_id=$MODEL_ID"
    fi
    if [ "$MODEL_SIZE" != "large" ]; then
        cmd="$cmd --patch experiments.0.run_config.model_spec.model_size=$MODEL_SIZE"
    fi
    
    echo "Command: $cmd"
    echo ""
    
    if [ -n "$SLURM_JOB_ID" ]; then
        srun $cmd
    else
        eval $cmd
    fi
}

# ==============================================================================
# Main
# ==============================================================================

# Parse arguments
BENCHMARKS=()
for arg in "$@"; do
    case $arg in
        -h|--help)
            show_help
            exit 0
            ;;
        all)
            BENCHMARKS=($ALL_BENCHMARKS)
            ;;
        beans|birdset|individual_id|vocal_repertoire)
            BENCHMARKS+=("$arg")
            ;;
        *)
            echo "ERROR: Unknown argument '$arg'"
            show_help
            exit 1
            ;;
    esac
done

# Default to all benchmarks if none specified
if [ ${#BENCHMARKS[@]} -eq 0 ]; then
    BENCHMARKS=($ALL_BENCHMARKS)
fi

# Print header
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  BEATs / OpenBEATs Evaluation                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Model ID:    $MODEL_ID"
echo "Model Size:  $MODEL_SIZE"
echo "Run Name:    $RUN_NAME"
echo "Config:      $EVAL_CONFIG"
echo "Benchmarks:  ${BENCHMARKS[*]}"
echo "Start Time:  $(date)"
echo ""

# Run evaluations
for bench in "${BENCHMARKS[@]}"; do
    run_eval "$bench"
done

# Print summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Evaluation Complete!"
echo "════════════════════════════════════════════════════════════════"
echo "Results saved to: evaluation_results/${RUN_NAME}_*"
echo "End Time: $(date)"
echo ""
