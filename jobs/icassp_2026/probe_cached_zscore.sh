#!/usr/bin/env bash
# Re-probe the cached pooled EAT embeddings (from job 91700) with raw AND z-scored
# features, 3 seeds each. CPU only -- no extraction, the h5 caches already exist.
#
# Why: the paper-spec probe underfits EAT's blocks.11.attn.proj features (EAT-all
# SSL: 49% TRAIN accuracy on Dogs). Z-scoring fixed it on Dogs/Watkins in a
# dev-VM test. This produces the full 6-dataset x 3-model table.
#
#   task  model
#    1    EAT-all SSL
#    2    sl-EAT-all published
#    3    sl-EAT-all REPRO (ours)
#
# Output: evaluation_results/icassp2026_<model>_probe_attnproj_pooled/zscore_probe.csv
# Submit:  sbatch jobs/icassp_2026/probe_cached_zscore.sh

#SBATCH --partition=cpu
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output="/home/%u/logs/zprobe-eat-%A_%a.log"
#SBATCH --job-name="zprobe-eat"

set -euo pipefail

cd ~/avex
uv sync --group project-dev

case "$SLURM_ARRAY_TASK_ID" in
  1) M=eat_ssl_all ;;
  2) M=eat_published_sl_all ;;
  3) M=eat_repro_sl_all ;;
  *) echo "ERROR: unexpected array task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
D=evaluation_results/icassp2026_${M}_probe_attnproj_pooled

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${M} ==="
uv run --group project-dev python scripts/probe_cached_embeddings.py \
    --cache-dir "$D" \
    --layer backbone.model.blocks.11.attn.proj \
    --seeds 0 1 2 \
    --threads 8 \
    --out "$D/zscore_probe.csv"
