#!/usr/bin/env bash

#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="clap-eval-beanspro"
#SBATCH --qos=naturelm

cd ~/avex
uv sync --group project-dev --group gpu

export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Latest best_model.pt under runs/clap. Runs both description splits.
srun uv run python scripts/eval_clap_beans_pro.py \
    --runs-root ~/avex/runs/clap \
    --split crow-description \
    --batch-size 32 \
    --num-workers 12 \
    --output ~/logs/beanspro_crow_${SLURM_JOB_ID}.json \
    --per-item-output ~/logs/beanspro_crow_${SLURM_JOB_ID}.per_item.jsonl

srun uv run python scripts/eval_clap_beans_pro.py \
    --runs-root ~/avex/runs/clap \
    --split zebra-description \
    --batch-size 32 \
    --num-workers 12 \
    --output ~/logs/beanspro_zebra_${SLURM_JOB_ID}.json \
    --per-item-output ~/logs/beanspro_zebra_${SLURM_JOB_ID}.per_item.jsonl
