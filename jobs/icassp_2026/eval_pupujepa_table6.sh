#!/usr/bin/env bash
# PupuJEPA on BEANS classification (AVEX Table 6: Probe / R-AUC / NMI), with the
# eval protocol that reproduced the paper's EAT rows:
#   - 32 kHz BEANS (model resamples to its native 24 kHz)
#   - pooled (N, 768) embedding cache (probe_storage_aggregation: mean)
#   - R-AUC + NMI from avex evaluate
#   - Probe = linear probe on z-scored features, paper spec (900 ep, lr 1e-4,
#     wd 0.1, bs 32), mean over 3 seeds -- scripts/probe_cached_embeddings.py,
#     run on the GPU right after extraction, reusing the cache (no re-extraction)
#
# MODEL=bigrun   PupuJEPA SSL (bio-jepa step-200000)
# MODEL=avexall  PupuJEPA post-trained on the AVEX 5-source mix
#                (needs runs/icassp2026/sl_pupujepa_avex_all_32k/backbone_final.safetensors,
#                 written by export_pupujepa_avexall.sh)
#
#   task  layer
#    1    final           (teacher output after final norm)
#    2    last_attn_proj  (teacher.blocks[-1].attn.proj -- the EAT-equivalent layer)
#
# Submit:  sbatch --export=ALL,MODEL=bigrun jobs/icassp_2026/eval_pupujepa_table6.sh
# (overnight_pupujepa.sh submits both with the right dependencies.)

#SBATCH --partition=h100-80
#SBATCH --array=1-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output="/home/%u/logs/%x-%A_%a.log"
#SBATCH --job-name="t6-pupujepa"

set -euo pipefail

: "${MODEL:?set MODEL=bigrun or MODEL=avexall}"
case "$MODEL" in bigrun|avexall) ;; *) echo "ERROR: MODEL must be bigrun or avexall, got '$MODEL'" >&2; exit 1 ;; esac

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
export AVEX_EVAL_WORKER_TIMEOUT_SECONDS=1800   # slow GCS reads (job 91649_1 died at 120s)

case "$SLURM_ARRAY_TASK_ID" in
  1) LAYER=final ;;
  2) LAYER=attnproj ;;
  *) echo "ERROR: unexpected array task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
TAG=pupujepa_${MODEL}_${LAYER}
CFG=configs/evaluation_configs/icassp_2026/${TAG}_eval.yml
OUT=evaluation_results/icassp2026_${TAG}

if [[ "$MODEL" == avexall && ! -f runs/icassp2026/sl_pupujepa_avex_all_32k/backbone_final.safetensors ]]; then
    echo "ERROR: post-trained backbone not exported yet (run export_pupujepa_avexall.sh)" >&2; exit 1
fi
if [[ -e "$OUT" ]]; then
    echo "ERROR: $OUT already exists -- refusing to overwrite" >&2; exit 1
fi

echo "=== ${TAG}: extract + retrieval + clustering + raw probe ==="
srun uv run avex evaluate --config "${CFG}"

echo "=== ${TAG}: z-scored probe on cached embeddings (GPU) ==="
srun uv run --group project-dev python scripts/probe_cached_embeddings.py \
    --cache-dir "$OUT" --layer output_features \
    --device cuda --variants zscore --seeds 0 1 2 \
    --out "$OUT/zscore_probe.csv"

echo "=== ${TAG}: done ==="
