#!/usr/bin/env bash
# Evaluate PupuJEPA-Base SSL (frozen, no post-training) across the AVEX benchmark.
# Array task 1 = probe (Table 6 accuracy), 2 = retrieval + clustering (R-AUC, NMI).
# Submit:  sbatch jobs/icassp_2026/eval_eat_ssl_all.sh
#
# Uses benchmark_beans_gcs_classification.yml: PupuJEPA runs at its native 24 kHz, following
# the paper's own treatment of Perch (32 kHz) and BirdNet (48 kHz) in S3.3.

#SBATCH --partition=h100-80
#SBATCH --array=1-2%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --output="/home/%u/logs/%A_%a.log"
#SBATCH --job-name="eat-eval"


cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

case "$SLURM_ARRAY_TASK_ID" in
  1) CFG=configs/evaluation_configs/icassp_2026/eat_ssl_all_probe.yml;     TAG=probe ;;
  2) CFG=configs/evaluation_configs/icassp_2026/eat_ssl_all_retrieval.yml; TAG=retrieval ;;
esac

srun uv run avex evaluate \
    --config "${CFG}" \
    --patch "dataset_config=configs/data_configs/benchmark_beans_gcs_classification.yml" \
    --patch "save_dir=evaluation_results/icassp2026_eat_ssl_all_${TAG}" \
    --patch "results_csv_path=evaluation_results/icassp2026_eat_ssl_all_${TAG}/results.csv"
