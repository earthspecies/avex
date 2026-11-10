#!/usr/bin/env bash
# BEANS linear probe (paper spec: 900 epochs, lr 1e-4, wd 0.1, bs 32) for the three
# EAT checkpoints on backbone.model.blocks.11.attn.proj -- the layer the retrieval
# ablation (job 91665) showed reproduces the paper's EAT-all row. Replaces the
# mlp.fc2 probes from eval_eat_repro_table.sh (tasks 1/3/5).
#
#   task  model
#    1    EAT-all SSL
#    2    sl-EAT-all published
#    3    sl-EAT-all REPRO (ours)
#
# The configs store MEAN-POOLED (N, 768) embeddings (probe_storage_aggregation:
# mean). Job 91675 ran without it: every probe was a linear layer on 393,984
# flattened features, not the paper's time-averaged embedding. Check the log for
# "Layer 'backbone.model.blocks.11.attn.proj' shape: torch.Size([N, 768])" --
# a 3-D shape means the fix did not apply.
#
# Submit:  sbatch jobs/icassp_2026/eval_eat_probe_attnproj.sh

#SBATCH --partition=h100-80
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output="/home/%u/logs/eatprobe-pooled-%A_%a.log"
#SBATCH --job-name="pooled-eatprobe"

set -euo pipefail

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
# Eval DataLoader per-batch timeout, raised from avex's 120s: job 91649_1 died
# after 2.5 h when one GCS read at CBI took longer than 120s.
export AVEX_EVAL_WORKER_TIMEOUT_SECONDS=1800

E=configs/evaluation_configs/icassp_2026
case "$SLURM_ARRAY_TASK_ID" in
  1) TAG=eat_ssl_all_probe_attnproj ;;
  2) TAG=eat_published_sl_all_probe_attnproj ;;
  3) TAG=eat_repro_sl_all_probe_attnproj ;;
  *) echo "ERROR: unexpected array task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
CFG=$E/${TAG}.yml

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${TAG} ==="

srun uv run avex evaluate \
    --config "${CFG}" \
    --patch "dataset_config=configs/data_configs/benchmark_beans_gcs_classification.yml" \
    --patch "save_dir=evaluation_results/icassp2026_${TAG}_pooled" \
    --patch "results_csv_path=evaluation_results/icassp2026_${TAG}_pooled/results.csv"
