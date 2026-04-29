#!/usr/bin/env bash

#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="clap-eval-cbi"
#SBATCH --qos=naturelm

cd ~/avex
uv sync --group project-dev --group gpu

export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Auto-finds the most recent best_model.pt under runs/clap.
# Override with --checkpoint <path> --config <path> if you want a specific one.
srun uv run python scripts/eval_clap_cbi.py \
    --runs-root ~/avex/runs/clap \
    --batch-size 64 \
    --num-workers 16 \
    --output ~/logs/cbi_zeroshot_${SLURM_JOB_ID}.json
