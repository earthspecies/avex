#!/usr/bin/env bash
# ==============================================================================
# OpenBEATs Large i3 Evaluation Script
# ==============================================================================
# Evaluates OpenBEATs Large i3 (best performing checkpoint) on all benchmarks:
#
#   1. beans            - BEANS Classification + Detection
#   2. birdset          - BirdSet Detection  
#   3. individual_id    - Individual ID Classification
#   4. vocal_repertoire - Vocal Repertoire Classification
#
# Usage:
#   ./jobs/openbeats_full_eval.sh                    # Run all benchmarks
#   ./jobs/openbeats_full_eval.sh beans              # Run BEANS only
#   ./jobs/openbeats_full_eval.sh birdset            # Run BirdSet only
#   ./jobs/openbeats_full_eval.sh individual_id      # Run Individual ID only
#   ./jobs/openbeats_full_eval.sh vocal_repertoire   # Run Vocal Repertoire only
#
# SLURM:
#   sbatch jobs/openbeats_full_eval.sh [benchmark]
# ==============================================================================

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --output="/home/%u/logs/openbeats_eval_%A.log"
#SBATCH --job-name="openbeats-eval"
#SBATCH --cpus-per-gpu=12

set -e

# Navigate to repo root
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
cd ~/representation-learning 2>/dev/null || cd "$(git rev-parse --show-toplevel)"

# Environment setup
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export BEANS_DEBUG=0

uv sync

# ==============================================================================
# Configuration
# ==============================================================================

EVAL_CONFIG="configs/evaluation_configs/openbeats_full_eval.yml"

declare -A DATA_CONFIGS=(
    ["beans"]="configs/data_configs/beans.yml"
    ["birdset"]="configs/data_configs/benchmark_birdset.yml"
    ["individual_id"]="configs/data_configs/individual_id_icassp.yml"
    ["vocal_repertoire"]="configs/data_configs/benchmark_id_repertoire_with_clustering.yml"
)

ALL_BENCHMARKS="beans birdset individual_id vocal_repertoire"

# ==============================================================================
# Main
# ==============================================================================

run_eval() {
    local bench=$1
    local data_cfg=${DATA_CONFIGS[$bench]}
    
    echo "════════════════════════════════════════"
    echo "  OpenBEATs Large i3 → $bench"
    echo "════════════════════════════════════════"
    
    local cmd="uv run repr-learn evaluate \
        --config $EVAL_CONFIG \
        --patch dataset_config=$data_cfg \
        --patch save_dir=evaluation_results/openbeats_$bench"
    
    if [ -n "$SLURM_JOB_ID" ]; then
        srun $cmd
    else
        $cmd
    fi
}

BENCHMARK=${1:-all}

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  OpenBEATs Large i3 Evaluation           ║"
echo "╚══════════════════════════════════════════╝"
echo "Benchmark: $BENCHMARK"
echo "Start: $(date)"
echo ""

case $BENCHMARK in
    all)
        for bench in $ALL_BENCHMARKS; do
            run_eval "$bench"
        done
        ;;
    beans|birdset|individual_id|vocal_repertoire)
        run_eval "$BENCHMARK"
        ;;
    *)
        echo "Usage: $0 [all|beans|birdset|individual_id|vocal_repertoire]"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════"
echo "Done! Results: evaluation_results/"
echo "End: $(date)"
echo "════════════════════════════════════════"
