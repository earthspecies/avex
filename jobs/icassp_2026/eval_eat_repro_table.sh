#!/usr/bin/env bash
# The sl-EAT-all reproduction table in ONE job: BEANS classification (Watkins,
# CBI, HBDB, BATS, Dogs, ESC-50) for three EAT checkpoints, all on the identical
# eval pipeline, so the rows are directly comparable to each other and to the
# AVEX paper's Table 1.
#
#   task  model                                   mode
#    1    EAT-all SSL          (before, published)  probe
#    2    EAT-all SSL          (before, published)  retrieval + clustering
#    3    sl-EAT-all           (target, published)  probe
#    4    sl-EAT-all           (target, published)  retrieval + clustering
#    5    sl-EAT-all REPRO     (ours)               probe
#    6    sl-EAT-all REPRO     (ours)               retrieval + clustering
#
# Rows 3-4 check the EVAL: they should land near the paper's sl-EAT-all row.
# Rows 5-6 check the TRAINING: they should land near rows 3-4.
#
# All six read the same 16 kHz BEANS config and probe the same encoder layer
# (backbone.model.blocks.11.mlp.fc2, mean-pooled), with the paper's probe spec
# (900 epochs, lr 1e-4, wd 0.1, bs 32). Tasks run concurrently, 1 GPU each; the
# 900-epoch probes (1, 3, 5) are the slow ones.
#
# Our checkpoint: runs/icassp2026/sl_eat_all_repro/2026-09-26_09-24-00/final_model.pt
# (verified: backbone identical to EAT-all SSL at epoch 2, 150/150 tensors changed
# by the end, 4.03% drift).
#
# CHECK each task's log for "Checkpoint loaded: N/M params matched" -- the eval
# loader is strict=False, so N must be close to M.
#
# Submit:  sbatch jobs/icassp_2026/eval_eat_repro_table.sh

#SBATCH --partition=h100-80
#SBATCH --array=1-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output="/home/%u/logs/eatrepro-eval-%A_%a.log"
#SBATCH --job-name="eatrepro-eval"

set -euo pipefail

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp

E=configs/evaluation_configs/icassp_2026
case "$SLURM_ARRAY_TASK_ID" in
  1) CFG=$E/eat_ssl_all_probe.yml;               TAG=eat_ssl_all_probe ;;
  2) CFG=$E/eat_ssl_all_retrieval.yml;           TAG=eat_ssl_all_retrieval ;;
  3) CFG=$E/eat_published_sl_all_probe.yml;      TAG=eat_published_sl_all_probe ;;
  4) CFG=$E/eat_published_sl_all_retrieval.yml;  TAG=eat_published_sl_all_retrieval ;;
  5) CFG=$E/eat_repro_sl_all_probe.yml;          TAG=eat_repro_sl_all_probe ;;
  6) CFG=$E/eat_repro_sl_all_retrieval.yml;      TAG=eat_repro_sl_all_retrieval ;;
  *) echo "ERROR: unexpected array task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${TAG} ==="

srun uv run avex evaluate \
    --config "${CFG}" \
    --patch "dataset_config=configs/data_configs/benchmark_beans_gcs_classification.yml" \
    --patch "save_dir=evaluation_results/icassp2026_${TAG}" \
    --patch "results_csv_path=evaluation_results/icassp2026_${TAG}/results.csv"
