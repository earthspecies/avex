#!/usr/bin/env bash
# Benchmark the POST-TRAINED PuPuJEPA (sl-PupuJEPA-all) on BEANS classification,
# to produce the sl-PupuJEPA-all row of Table 1 alongside the existing
# PupuJEPA-all^SSL row.
#
# Checkpoint: runs/icassp2026/sl_pupujepa_audioset_xeno_inat/2026-08-31_23-05-45
# (10 epochs supervised on AudioSet + Xeno-canto + iNaturalist, 7 GPUs, batch 224).
#
# This MIRRORS eval_pupujepa_bigrun.sh exactly -- same two tasks, same dataset
# config, same patches -- so the numbers land in the same table. The only
# difference is the eval config, which differs from the SSL one solely in
# checkpoint_path. That SSL job reproduces the paper's PupuJEPA-all row to 3
# decimals on all 18 numbers, so this is a like-for-like comparison.
#
#   task 1  probe                   -> Probe column  (linear probe on frozen
#                                      final-layer embeddings, 900 epochs)
#   task 2  retrieval + clustering  -> R-AUC and NMI columns (no probe trained)
#
# Six BEANS datasets, reported as: Watkins = marine_mammal, CBI = bird,
# HBDB = mosquito, BATS = bat, Dogs = dog, ESC-50 = esc50.
#
# Results: evaluation_results/icassp2026_pupujepa_posttrained_<tag>/results.csv
#
# Submit:  sbatch jobs/icassp_2026/eval_pupujepa_posttrained.sh
#
# No --exclude: excluding a node just makes the job queue behind idle hardware.

#SBATCH --partition=h100-80
#SBATCH --array=1-2%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output="/home/%u/logs/pupupost-%A_%a.log"
#SBATCH --job-name="pupupost-eval"

set -euo pipefail

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp

CKPT=runs/icassp2026/sl_pupujepa_audioset_xeno_inat/2026-08-31_23-05-45/backbone_final.safetensors
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: post-trained backbone not found at $CKPT" >&2
    exit 1
fi

case "$SLURM_ARRAY_TASK_ID" in
  1) CFG=configs/evaluation_configs/icassp_2026/pupujepa_posttrained_probe.yml;     TAG=probe ;;
  2) CFG=configs/evaluation_configs/icassp_2026/pupujepa_posttrained_retrieval.yml; TAG=retrieval ;;
  *) echo "ERROR: unexpected array task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${TAG} (BEANS classification, 32k) ==="

srun uv run avex evaluate \
    --config "${CFG}" \
    --patch "dataset_config=configs/data_configs/benchmark_beans_gcs_32k_classification.yml" \
    --patch "save_dir=evaluation_results/icassp2026_pupujepa_posttrained_${TAG}" \
    --patch "results_csv_path=evaluation_results/icassp2026_pupujepa_posttrained_${TAG}/results.csv"
