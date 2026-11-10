#!/usr/bin/env bash
# Evaluate Marius's existing 32 kHz EfficientNet-B0 (effnet_32khz_v0) on BEANS
# classification: task 1 probe, task 2 retrieval + clustering. Same protocol as the
# PupuJEPA/EAT evals. Checkpoint is on GCS (gs://representation-learning/models/v1/
# effnet_32khz_v0/final_model.pt); if the job dies with FileNotFoundError it has moved.
# CHECK the log for "Checkpoint loaded: N/M params matched" -- the eval loader is
# strict=False, so a key mismatch would silently probe a random backbone.
#
# Submit:  sbatch jobs/icassp_2026/eval_effnet_32khz_v0.sh

#SBATCH --partition=h100-80
#SBATCH --array=1-2%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output="/home/%u/logs/effnet32-%A_%a.log"
#SBATCH --job-name="effnet32-eval"

set -euo pipefail

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp


case "$SLURM_ARRAY_TASK_ID" in
  1) CFG=configs/evaluation_configs/icassp_2026/effnet_32khz_v0_probe.yml;     TAG=probe ;;
  2) CFG=configs/evaluation_configs/icassp_2026/effnet_32khz_v0_retrieval.yml; TAG=retrieval ;;
  *) echo "ERROR: unexpected array task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${TAG} (BEANS classification, 32k, effnet_32khz_v0) ==="

srun uv run avex evaluate \
    --config "${CFG}" \
    --patch "dataset_config=configs/data_configs/benchmark_beans_gcs_32k_classification.yml" \
    --patch "save_dir=evaluation_results/icassp2026_effnet_32khz_v0_${TAG}" \
    --patch "results_csv_path=evaluation_results/icassp2026_effnet_32khz_v0_${TAG}/results.csv"
