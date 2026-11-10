#!/usr/bin/env bash
# XC_1M MAE (Moritz) on AVEX Tables 6-9, with the protocol used for EAT and PupuJEPA:
# pooled cache, R-AUC / NMI from avex evaluate, Probe = z-scored linear probe (paper
# spec, 3 seeds) on the cached embeddings. Configs: configs/evaluation_configs/
# icassp_2026/xenomae_<MODEL>_t{6,7,8,9}_eval.yml (generated from the PupuJEPA ones).
#
#   MODEL=xc1m      SSL checkpoint (/home/.../moritz_mvp/model.safetensors), mean pooling
#   MODEL=avexall   post-trained (runs/icassp2026/sl_xenomae_avex_all_32k/backbone_final.safetensors)
#   MODEL=xc1m_cls  ablation: SSL with CLS pooling, Table 6 retrieval + clustering only (--array=1)
#
#   task  table
#    1    6  BEANS classification
#    2    7  BEANS detection
#    3    8  BirdSet
#    4    9  Individual ID + Vocal Repertoire
#
# Submit:  sbatch --export=ALL,MODEL=xc1m --job-name=xmae-ssl jobs/icassp_2026/eval_xenomae_tables.sh
# (overnight_xenomae.sh submits all of them with the right dependencies.)

#SBATCH --partition=h100-80
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output="/home/%u/logs/%x-%A_%a.log"
#SBATCH --job-name="xmae-eval"

set -euo pipefail

: "${MODEL:?set MODEL=xc1m, avexall or xc1m_cls}"
case "$MODEL" in xc1m|avexall|xc1m_cls) ;; *) echo "ERROR: bad MODEL '$MODEL'" >&2; exit 1 ;; esac

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
[[ "$MODEL" == xc1m_cls && "$TB" != t6 ]] && { echo "xc1m_cls is a Table 6 ablation only"; exit 0; }

TAG=xenomae_${MODEL}_${TB}
CACHE=xenomae_${MODEL}          # run_config run_name -> cache dirs <dataset>_${CACHE}
CFG=configs/evaluation_configs/icassp_2026/${TAG}_eval.yml
OUT=evaluation_results/icassp2026_${TAG}

if [[ "$MODEL" == avexall && ! -f runs/icassp2026/sl_xenomae_avex_all_32k/backbone_final.safetensors ]]; then
    echo "ERROR: post-trained backbone not exported yet" >&2; exit 1
fi
if [[ -e "$OUT" ]]; then
    echo "ERROR: $OUT already exists -- refusing to overwrite (move it aside)" >&2; exit 1
fi

echo "=== ${TAG}: avex evaluate ==="
# env binaries directly: inside srun the cluster resets UV_* to the full /scratch (job 94467)
srun "$UV_PROJECT_ENVIRONMENT/bin/avex" evaluate --config "${CFG}"

if [[ "$MODEL" != xc1m_cls ]]; then
    echo "=== ${TAG}: z-scored probe on cached embeddings (GPU) ==="
    srun "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/probe_cached_embeddings.py \
        --cache-dir "$OUT" --layer output_features --run-name "$CACHE" \
        --device cuda --variants zscore --seeds 0 1 2 \
        --out "$OUT/zscore_probe.csv"
fi
echo "=== ${TAG}: done ==="
