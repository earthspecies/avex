#!/usr/bin/env bash
# (b) Evaluate eat_ssl_all (esp-aves2-eat-all, hf://) by retrieval + clustering across
#     the AVEX benchmark, one dataset per array task. Results land under
#     evaluation_results/icassp2026_eat_ssl_all_<tag>/results.csv (aggregate with
#     ~/agg_results.py). Submit:  sbatch jobs/icassp_2026/eval_eat_ssl_all_retrieval.sh

#SBATCH --partition=h100-80
#SBATCH --array=1-4%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --output="/home/%u/logs/%A_%a.log"
#SBATCH --job-name="eat-retr"
#SBATCH --qos=naturelm

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

case "$SLURM_ARRAY_TASK_ID" in
  1) DS=configs/data_configs/benchmark_beans_gcs.yml;           TAG=beans ;;
  2) DS=configs/data_configs/benchmark_birdset_gs.yml;          TAG=birdset ;;
  3) DS=configs/data_configs/benchmark_id_repertoire_48khz.yml; TAG=repertoire ;;
  4) DS=configs/data_configs/individual_id_gcs.yml;             TAG=individual_id ;;
esac

srun uv run avex evaluate \
    --config configs/evaluation_configs/icassp_2026/eat_ssl_all_retrieval.yml \
    --patch "dataset_config=${DS}" \
    --patch "save_dir=evaluation_results/icassp2026_eat_ssl_all_${TAG}" \
    --patch "results_csv_path=evaluation_results/icassp2026_eat_ssl_all_${TAG}/results.csv"
