#!/usr/bin/env bash
# AVEX Table 9 (Individual ID + Vocal Repertoire) for the four models of Tables 6-8.
#   Probe  = z-scored linear probe accuracy (paper spec, 3 seeds) on *_across_year / macaques
#   R-AUC* = retrieval with TRAIN items as queries (*_train_vs_test sets, paper S3.4)
#   R-AUC, NMI = retrieval / k-means clustering on the test split (call-type sets)
# Data: configs/data_configs/benchmark_table9_{16k,32k}.yml (ESP's individual_id_icassp.yml
# entries + Bengalese Finch bird0). SRKW is not included -- its dataset is not
# available (see the config header).
#
#   task  model                       rate
#    1    EAT-all SSL                 16k
#    2    sl-EAT-all (our post-train) 16k
#    3    PupuJEPA SSL (final layer)  32k
#    4    sl-PupuJEPA-all (final)     32k
#
# The datasets are small (324-7,652 clips each), so expect well under the time limit.
#
# Submit:  sbatch jobs/icassp_2026/eval_table9.sh

#SBATCH --partition=h100-80
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=12:00:00
#SBATCH --output="/home/%u/logs/t9-%A_%a.log"
#SBATCH --job-name="t9-eval"

set -euo pipefail

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
export AVEX_EVAL_WORKER_TIMEOUT_SECONDS=1800   # slow GCS reads (job 91649_1 died at 120s)

# CACHE = the run_config's run_name: avex names each dataset's cache dir <dataset>_<CACHE>.
case "$SLURM_ARRAY_TASK_ID" in
  1) TAG=eat_ssl_all_t9;      CACHE=eat_ssl_all;      LAYER=backbone.model.blocks.11.attn.proj ;;
  2) TAG=eat_repro_sl_all_t9; CACHE=eat_ssl_all;      LAYER=backbone.model.blocks.11.attn.proj ;;
  3) TAG=pupujepa_bigrun_t9;  CACHE=pupujepa_bigrun;  LAYER=output_features ;;
  4) TAG=pupujepa_avexall_t9; CACHE=pupujepa_avexall; LAYER=output_features ;;
  *) echo "ERROR: unexpected array task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
CFG=configs/evaluation_configs/icassp_2026/${TAG}_eval.yml
OUT=evaluation_results/icassp2026_${TAG}

if [[ -e "$OUT" ]]; then
    echo "ERROR: $OUT already exists -- refusing to overwrite" >&2; exit 1
fi

echo "=== ${TAG}: extract + probe + retrieval + clustering ==="
srun uv run avex evaluate --config "${CFG}"

echo "=== ${TAG}: z-scored probe on cached embeddings (GPU) ==="
srun uv run --group project-dev python scripts/probe_cached_embeddings.py \
    --cache-dir "$OUT" --layer "$LAYER" --run-name "$CACHE" \
    --device cuda --variants zscore --seeds 0 1 2 \
    --out "$OUT/zscore_probe.csv"

echo "=== ${TAG}: done ==="
