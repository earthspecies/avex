#!/usr/bin/env bash
# Evaluate the POST-TRAINED EAT checkpoint (supervised on AudioSet + Xeno-canto +
# iNaturalist, run 2026-08-25_16-42-16) on the AVEX benchmark.
#
# ONE array job, 5 tasks, NO overlap between them. Each task is a distinct
# (eval mode, dataset) pair and runs on its OWN GPU, all 5 CONCURRENTLY --
# SLURM schedules them across whatever nodes are free, so nothing is sequential:
#
#   1  probe                   BEANS (classification)   <- AVEX Table 6 Probe column
#   2  retrieval + clustering  BEANS
#   3  retrieval + clustering  BirdSet
#   4  retrieval + clustering  ID repertoire (48k)
#   5  retrieval + clustering  individual ID
#
# Task 1 is by far the longest (900 probe epochs); tasks 2-5 are embedding-based
# and quick. There is deliberately no %N throttle on the array, so all 5 start as
# soon as GPUs free up. Add %N (e.g. --array=1-5%2) if you need to be polite to
# other users.
#
# Results: evaluation_results/icassp2026_eat_posttrained_<tag>/results.csv
# (aggregate with ~/agg_results.py).
#
# Submit:  sbatch jobs/icassp_2026/eval_eat_posttrained.sh
#
# SANITY CHECK once running: the log must show "Checkpoint loaded: 150/152
# params matched". 150/152 = every backbone tensor loaded, only the 2 classifier
# tensors skipped (intentional -- probing uses the frozen backbone, not the head).
# A number near 0/152 means the keys did not reconcile and the backbone is
# RANDOM -- kill it, since strict=False will not raise on its own.
#
# No --qos: none of the working ESP jobs set one, and --qos=naturelm is rejected.

#SBATCH --partition=h100-80
#SBATCH --array=1-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output="/home/%u/logs/eatpost-%A_%a.log"
#SBATCH --job-name="eatpost-eval"

set -euo pipefail

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp

CKPT=runs/icassp2026/sl_eat_audioset_xeno_inat/2026-08-25_16-42-16/final_model.pt
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found at $CKPT" >&2
    exit 1
fi

PROBE=configs/evaluation_configs/icassp_2026/eat_posttrained_probe.yml
RETR=configs/evaluation_configs/icassp_2026/eat_posttrained_retrieval.yml

case "$SLURM_ARRAY_TASK_ID" in
  1) CFG=$PROBE; DS=configs/data_configs/benchmark_beans_gcs_classification.yml; TAG=probe_beans ;;
  2) CFG=$RETR;  DS=configs/data_configs/benchmark_beans_gcs.yml;                TAG=retr_beans ;;
  3) CFG=$RETR;  DS=configs/data_configs/benchmark_birdset_gs.yml;               TAG=retr_birdset ;;
  4) CFG=$RETR;  DS=configs/data_configs/benchmark_id_repertoire_48khz.yml;      TAG=retr_repertoire ;;
  5) CFG=$RETR;  DS=configs/data_configs/individual_id_gcs.yml;                  TAG=retr_individual_id ;;
  *) echo "ERROR: unexpected array task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${TAG} | config=${CFG} | dataset=${DS} ==="

srun uv run avex evaluate \
    --config "${CFG}" \
    --patch "dataset_config=${DS}" \
    --patch "save_dir=evaluation_results/icassp2026_eat_posttrained_${TAG}" \
    --patch "results_csv_path=evaluation_results/icassp2026_eat_posttrained_${TAG}/results.csv"
