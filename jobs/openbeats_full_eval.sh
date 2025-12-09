#!/usr/bin/env bash
# ==============================================================================
# OpenBEATs Full Evaluation Script
# ==============================================================================
# This script runs all evaluations from the paper on OpenBEATs pretrained models:
#
# Benchmarks (matching the paper table):
#   1. BEANS Classification - Probe accuracy, R-AUC, NMI
#   2. BEANS Detection - Probe (mAP), R-AUC  
#   3. BirdSet - Probe (mAP), R-AUC
#   4. Individual ID - Probe accuracy, R-AUC
#   5. Vocal Repertoire - R-AUC, NMI
#
# Usage:
#   ./jobs/openbeats_full_eval.sh                    # Run all benchmarks
#   ./jobs/openbeats_full_eval.sh beans              # Run BEANS only
#   ./jobs/openbeats_full_eval.sh birdset            # Run BirdSet only
#   ./jobs/openbeats_full_eval.sh individual_id      # Run Individual ID only
#   ./jobs/openbeats_full_eval.sh vocal_repertoire   # Run Vocal Repertoire only
#
# SLURM Usage:
#   sbatch jobs/openbeats_full_eval.sh
# ==============================================================================

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --output="/home/%u/logs/openbeats_eval_%A.log"
#SBATCH --job-name="openbeats-full-eval"
#SBATCH --cpus-per-gpu=12

set -e

# Navigate to repo root
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
cd ~/representation-learning 2>/dev/null || cd "$(git rev-parse --show-toplevel)"

# Environment setup
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export BEANS_DEBUG=0

# Sync dependencies
uv sync

# ==============================================================================
# Benchmark Configurations
# ==============================================================================
# Maps benchmark names to their data config files

declare -A BENCHMARK_CONFIGS=(
    ["beans"]="configs/data_configs/beans.yml"
    ["birdset"]="configs/data_configs/benchmark_birdset.yml"
    ["individual_id"]="configs/data_configs/individual_id_icassp.yml"
    ["vocal_repertoire"]="configs/data_configs/benchmark_id_repertoire_with_clustering.yml"
)

# Evaluation config for OpenBEATs
EVAL_CONFIG="configs/evaluation_configs/openbeats_full_eval.yml"

# ==============================================================================
# Helper Functions
# ==============================================================================

run_benchmark() {
    local benchmark_name=$1
    local data_config=${BENCHMARK_CONFIGS[$benchmark_name]}
    
    if [ -z "$data_config" ]; then
        echo "Error: Unknown benchmark '$benchmark_name'"
        echo "Available benchmarks: ${!BENCHMARK_CONFIGS[*]}"
        exit 1
    fi
    
    echo "========================================"
    echo "Running OpenBEATs evaluation on: $benchmark_name"
    echo "Data config: $data_config"
    echo "========================================"
    
    # Use srun if in SLURM environment, otherwise run directly
    if [ -n "$SLURM_JOB_ID" ]; then
        srun uv run repr-learn evaluate \
            --config "$EVAL_CONFIG" \
            --patch "dataset_config=$data_config" \
            --patch "save_dir=evaluation_results/openbeats_${benchmark_name}"
    else
        uv run repr-learn evaluate \
            --config "$EVAL_CONFIG" \
            --patch "dataset_config=$data_config" \
            --patch "save_dir=evaluation_results/openbeats_${benchmark_name}"
    fi
    
    echo "Completed: $benchmark_name"
    echo ""
}

# ==============================================================================
# Main Execution
# ==============================================================================

BENCHMARK=${1:-all}

echo "=============================================="
echo "OpenBEATs Full Evaluation"
echo "=============================================="
echo "Running benchmark(s): $BENCHMARK"
echo "Start time: $(date)"
echo ""

case $BENCHMARK in
    all)
        echo "Running ALL benchmarks from the paper..."
        echo ""
        for bench in beans birdset individual_id vocal_repertoire; do
            run_benchmark "$bench"
        done
        ;;
    beans|birdset|individual_id|vocal_repertoire)
        run_benchmark "$BENCHMARK"
        ;;
    *)
        echo "Error: Unknown benchmark '$BENCHMARK'"
        echo ""
        echo "Usage: $0 [benchmark]"
        echo ""
        echo "Available benchmarks:"
        echo "  all              - Run all benchmarks (default)"
        echo "  beans            - BEANS Classification + Detection"
        echo "  birdset          - BirdSet Detection"
        echo "  individual_id    - Individual ID Classification"
        echo "  vocal_repertoire - Vocal Repertoire Classification"
        exit 1
        ;;
esac

echo "=============================================="
echo "All evaluations completed!"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Results saved to: evaluation_results/"
echo "Combined CSV: evaluation_results/openbeats_all_benchmarks.csv"
