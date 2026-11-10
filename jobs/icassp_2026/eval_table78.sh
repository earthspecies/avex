#!/usr/bin/env bash
# AVEX Table 7 (BEANS detection: enabirds, rfcx, hiceas, gibbons, dcase) and
# Table 8 (BirdSet: POW, PER, NES, NBP, HSN, SNE, UHH) -- Probe = mAP, R-AUC --
# for the four models already validated on Table 6, with the same protocol:
#   pooled (N, 768) cache, R-AUC from avex evaluate, Probe = z-scored linear probe
#   (paper spec: 900 ep, lr 1e-4, wd 0.1, bs 32), multi-label BCE, best epoch by
#   val mAP, mean over 3 seeds (scripts/probe_cached_embeddings.py, GPU).
#
#   task  model                       rate  table
#    1    EAT-all SSL                 16k   7
#    2    EAT-all SSL                 16k   8
#    3    sl-EAT-all (our post-train) 16k   7
#    4    sl-EAT-all (our post-train) 16k   8
#    5    PupuJEPA SSL (final layer)  32k   7
#    6    PupuJEPA SSL (final layer)  32k   8
#    7    sl-PupuJEPA-all (final)     32k   7
#    8    sl-PupuJEPA-all (final)     32k   8
#
# Rows 1-4 check the detection pipeline against the paper's EAT-all / sl-EAT-all
# rows before rows 5-8 are trusted.
#
# Depends on three fixes (all verified locally):
#   - avex/evaluation/embedding_manager.py: offline probe num_labels = label width
#     (it was ~1 for multi-label targets -> CUDA assert)
#   - avex/data/transforms.py: stratified train_val_split on list-valued labels
#   - avex/data/birdset_train_splits.py: read pre-resampled 16k/32k train audio
#
# Submit:  sbatch jobs/icassp_2026/eval_table78.sh

#SBATCH --partition=h100-80
#SBATCH --array=1-8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output="/home/%u/logs/t78-%A_%a.log"
#SBATCH --job-name="t78-eval"

set -euo pipefail

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
export AVEX_EVAL_WORKER_TIMEOUT_SECONDS=1800   # slow GCS reads (job 91649_1 died at 120s)

case "$SLURM_ARRAY_TASK_ID" in
  1) TAG=eat_ssl_all_t7 ;;       2) TAG=eat_ssl_all_t8 ;;
  3) TAG=eat_repro_sl_all_t7 ;;  4) TAG=eat_repro_sl_all_t8 ;;
  5) TAG=pupujepa_bigrun_t7 ;;   6) TAG=pupujepa_bigrun_t8 ;;
  7) TAG=pupujepa_avexall_t7 ;;  8) TAG=pupujepa_avexall_t8 ;;
  *) echo "ERROR: unexpected array task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
case "$TAG" in
  eat_*)      LAYER=backbone.model.blocks.11.attn.proj ;;
  pupujepa_*) LAYER=output_features ;;
esac
CFG=configs/evaluation_configs/icassp_2026/${TAG}_eval.yml
OUT=evaluation_results/icassp2026_${TAG}

if [[ -e "$OUT" ]]; then
    echo "ERROR: $OUT already exists -- refusing to overwrite" >&2; exit 1
fi

echo "=== ${TAG}: extract + retrieval + raw probe ==="
srun uv run avex evaluate --config "${CFG}"

echo "=== ${TAG}: z-scored multi-label probe on cached embeddings (GPU) ==="
srun uv run --group project-dev python scripts/probe_cached_embeddings.py \
    --cache-dir "$OUT" --layer "$LAYER" \
    --device cuda --variants zscore --seeds 0 1 2 \
    --out "$OUT/zscore_probe.csv"

echo "=== ${TAG}: done ==="
