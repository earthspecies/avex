#!/usr/bin/env bash
# EfficientNet-B0 32 kHz (Marius's effnet_32khz_v0) on AVEX Tables 6-9 -- same protocol and
# script as eval_xenomae_tables.sh (copied; only names, layer and model choice differ).
#
#   MODEL=ssl      Marius's checkpoint as handed over
#   MODEL=avexall  post-trained (runs/icassp2026/sl_effnet32k_avex_all/final_model.pt)
#
#   task  table:  1 = 6 BEANS cls,  2 = 7 BEANS det,  3 = 8 BirdSet,  4 = 9 ID + repertoire
#
# Embedding: model.features.8.0 (final 1280-ch conv), avex's EfficientNet "mean" pooling.
# Submit:  sbatch --export=ALL,MODEL=ssl --job-name=eff-ssl jobs/icassp_2026/eval_effnet_tables.sh

#SBATCH --partition=h100-80
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output="/home/%u/logs/%x-%A_%a.log"
#SBATCH --job-name="eff-eval"

set -euo pipefail

: "${MODEL:?set MODEL=ssl or avexall}"
case "$MODEL" in ssl|avexall) ;; *) echo "ERROR: bad MODEL '$MODEL'" >&2; exit 1 ;; esac

# The cluster points uv at the node-local /scratch, which is full on several nodes
# (jobs 94459/94462: "No space left on device" during uv sync). Use home instead:
# shared wheel cache, and a per-job environment removed when the job ends.
export UV_CACHE_DIR="$HOME/cache/uv"
export UV_PROJECT_ENVIRONMENT="$HOME/cache/jobenvs/avex-${SLURM_JOB_ID}"
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT"' EXIT

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
export AVEX_EVAL_WORKER_TIMEOUT_SECONDS=1800

case "$SLURM_ARRAY_TASK_ID" in
  1) TB=t6 ;; 2) TB=t7 ;; 3) TB=t8 ;; 4) TB=t9 ;;
  *) echo "ERROR: unexpected array task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

TAG=effnet32k_${MODEL}_${TB}
CACHE=effnet_32khz_v0          # run_config run_name -> cache dirs <dataset>_${CACHE}
CFG=configs/evaluation_configs/icassp_2026/${TAG}_eval.yml
OUT=evaluation_results/icassp2026_${TAG}

if [[ "$MODEL" == avexall && ! -f runs/icassp2026/sl_effnet32k_avex_all/final_model.pt ]]; then
    echo "ERROR: post-trained checkpoint not copied to runs/icassp2026/sl_effnet32k_avex_all/final_model.pt yet" >&2; exit 1
fi
if [[ -e "$OUT" ]]; then
    echo "ERROR: $OUT already exists -- refusing to overwrite (move it aside)" >&2; exit 1
fi

echo "=== ${TAG}: avex evaluate ==="
# env binaries directly: inside srun the cluster resets UV_* to the full /scratch (job 94467)
srun "$UV_PROJECT_ENVIRONMENT/bin/avex" evaluate --config "${CFG}"

if true; then
    echo "=== ${TAG}: z-scored probe on cached embeddings (GPU) ==="
    srun "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/probe_cached_embeddings.py \
        --cache-dir "$OUT" --layer model.features.8.0 --run-name "$CACHE" \
        --device cuda --variants zscore --seeds 0 1 2 \
        --out "$OUT/zscore_probe.csv"
fi
echo "=== ${TAG}: done ==="
