#!/usr/bin/env bash
# Evaluate PupuJEPA-Base SSL (frozen, no post-training) across the AVEX benchmark.
# Array task 1 = probe (Table 6 accuracy), 2 = retrieval + clustering (R-AUC, NMI).
# Submit:  sbatch jobs/icassp_2026/eval_pupujepa_bigrun.sh
#
# Uses benchmark_beans_gcs_32k_classification.yml: PupuJEPA runs at its native 24 kHz, following
# the paper's own treatment of Perch (32 kHz) and BirdNet (48 kHz) in S3.3.

#SBATCH --partition=h100-80
#SBATCH --array=1-2%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --output="/home/%u/logs/%A_%a.log"
#SBATCH --job-name="pupu-eval"
# slurm-8x-h100-4 has a full /scratch: `uv sync` fails partway extracting triton,
# then the run dies with a misleading `ModuleNotFoundError: No module named
# 'alp_data'`. It also threw the uncorrectable ECC error during pretraining.
# Remove this once its scratch has been cleaned.
#SBATCH --exclude=slurm-8x-h100-4

cd ~/avex
uv sync --group project-dev --group gpu
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

case "$SLURM_ARRAY_TASK_ID" in
  1) CFG=configs/evaluation_configs/icassp_2026/pupujepa_bigrun_probe.yml;     TAG=probe ;;
  2) CFG=configs/evaluation_configs/icassp_2026/pupujepa_bigrun_retrieval.yml; TAG=retrieval ;;
esac

srun uv run avex evaluate \
    --config "${CFG}" \
    --patch "dataset_config=configs/data_configs/benchmark_beans_gcs_32k_classification.yml" \
    --patch "save_dir=evaluation_results/icassp2026_pupujepa_bigrun_${TAG}" \
    --patch "results_csv_path=evaluation_results/icassp2026_pupujepa_bigrun_${TAG}/results.csv"
