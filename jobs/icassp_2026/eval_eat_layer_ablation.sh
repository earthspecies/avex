#!/usr/bin/env bash
# EAT eval-LAYER ablation, retrieval + clustering only (fast, ~25 min per task).
#
# Why: with target_layers = blocks.11.mlp.fc2 (eval_eat_repro_table.sh), the
# paper's own sl-EAT-all checkpoint (HF safetensors == gs://.../sl_eat_all_ssl_all.pt,
# bit-identical) scores R-AUC 0.632 / NMI 0.298 -- BELOW the SSL base (0.672 /
# 0.397), while the paper reports 0.791 / 0.536. So our eval is not the paper's.
# fc2 is only the FFN branch of the last block (before the residual add and the
# post-norm), not "the final layer" the paper describes (S3.4). This tests the
# two alternatives on all three checkpoints; the layer that puts the published
# checkpoint near the paper is the protocol to use for the probe runs.
#
#   task  model                   layer
#    1    EAT-all SSL             blocks.11            (block output, post-norm)
#    2    EAT-all SSL             blocks.11.attn.proj  (what 'last_layer' resolves to)
#    3    sl-EAT-all published    blocks.11
#    4    sl-EAT-all published    blocks.11.attn.proj
#    5    sl-EAT-all REPRO        blocks.11
#    6    sl-EAT-all REPRO        blocks.11.attn.proj
#
# Each task gets its own save_dir: avex names the embedding cache after the
# run_config's run_name (shared by all three), and the clustering cache file
# ignores the layer, so shared dirs would mix embeddings.
#
# Submit:  sbatch jobs/icassp_2026/eval_eat_layer_ablation.sh

#SBATCH --partition=h100-80
#SBATCH --array=1-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=06:00:00
#SBATCH --output="/home/%u/logs/eatlayer-eval-%A_%a.log"
#SBATCH --job-name="eatlayer-eval"

set -euo pipefail

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp

E=configs/evaluation_configs/icassp_2026
case "$SLURM_ARRAY_TASK_ID" in
  1) TAG=eat_ssl_all_retrieval_block11 ;;
  2) TAG=eat_ssl_all_retrieval_attnproj ;;
  3) TAG=eat_published_sl_all_retrieval_block11 ;;
  4) TAG=eat_published_sl_all_retrieval_attnproj ;;
  5) TAG=eat_repro_sl_all_retrieval_block11 ;;
  6) TAG=eat_repro_sl_all_retrieval_attnproj ;;
  *) echo "ERROR: unexpected array task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
CFG=$E/${TAG}.yml

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${TAG} ==="

srun uv run avex evaluate \
    --config "${CFG}" \
    --patch "dataset_config=configs/data_configs/benchmark_beans_gcs_classification.yml" \
    --patch "save_dir=evaluation_results/icassp2026_${TAG}" \
    --patch "results_csv_path=evaluation_results/icassp2026_${TAG}/results.csv"
