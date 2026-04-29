#!/usr/bin/env bash

#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="biolingual-eval"
#SBATCH --qos=naturelm

cd ~/avex
uv sync --group project-dev --group gpu

export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Full splits (--limit 0 = no limit). Crow then zebra.
srun uv run python scripts/eval_biolingual_beans_pro.py \
    --split crow-description \
    --limit 0 \
    --output ~/logs/biolingual_crow_${SLURM_JOB_ID}.json \
    --per-item-output ~/logs/biolingual_crow_${SLURM_JOB_ID}.per_item.jsonl

srun uv run python scripts/eval_biolingual_beans_pro.py \
    --split zebra-description \
    --limit 0 \
    --output ~/logs/biolingual_zebra_${SLURM_JOB_ID}.json \
    --per-item-output ~/logs/biolingual_zebra_${SLURM_JOB_ID}.per_item.jsonl
