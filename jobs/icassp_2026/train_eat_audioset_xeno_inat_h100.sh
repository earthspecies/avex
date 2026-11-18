#!/usr/bin/env bash
# (a) Post-train EAT (SSL init) supervised on AudioSet + Xeno-canto + iNaturalist,
#     single H100. Joint multi-label space over {AudioSet classes} U {species}.
#     Submit:  sbatch jobs/icassp_2026/train_eat_audioset_xeno_inat_h100.sh

#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="eat-sl-train"
#SBATCH --qos=naturelm

cd ~/avex
uv sync --group project-dev --group gpu

srun uv run avex train \
    --config configs/run_configs/icassp_2026/sl_eat_audioset_xeno_inat.yml

# Smoke test instead (tiny subsets, no wandb, verifies configs build + step):
#   srun uv run avex train \
#       --config configs/run_configs/icassp_2026/sl_eat_audioset_xeno_inat_smoke.yml
