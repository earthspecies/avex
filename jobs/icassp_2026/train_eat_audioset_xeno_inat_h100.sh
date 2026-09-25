#!/usr/bin/env bash
# Post-train EAT (SSL init) supervised on AudioSet + Xeno-canto + iNaturalist.
# Joint multi-label space over {AudioSet classes} U {species}.
#
# LAUNCHER, not a batch script -- run it directly on the login node:
#     bash jobs/icassp_2026/train_eat_audioset_xeno_inat_h100.sh
#     NGPU=8 BS=16 bash jobs/icassp_2026/train_eat_audioset_xeno_inat_h100.sh
#     TIME=24:00:00 bash jobs/icassp_2026/train_eat_audioset_xeno_inat_h100.sh
#     RESUME=runs/icassp2026/sl_eat_audioset_xeno_inat/<timestamp> bash ...
# It generates the sbatch file (so NGPU can vary -- #SBATCH directives are
# static) and submits it. Same pattern as ddpsmoke_eat_multigpu.sh.
#
# DDP: avex only creates a process group when world_size > 1, and world_size
# comes from SLURM_NTASKS -- so this needs ONE TASK PER GPU. If the log says
# "world size 1" it is silently training on a single card.
#
# BS is PER-GPU (DataLoader + DistributedSampler), so global = BS * NGPU.
# The 4 x 32 default is the layout verified by the DDP smoke test and gives
# the global batch of 128 that lr=1e-4 was chosen for. 8 x 16 is the same
# global batch on twice the GPUs (and strictly less memory per card).
#
# SLURM layout follows bio-jepa/configs/slurm/h100_80_x8.yaml. Deliberately no
# --qos: none of the working ESP jobs set one, and --qos=naturelm is rejected.

set -euo pipefail

NGPU="${NGPU:-4}"           # h100-80 nodes have 8 GPUs total
BS="${BS:-32}"              # per-GPU batch size; global = BS * NGPU
CPUS="${CPUS:-24}"          # cpus PER TASK; NGPU*CPUS must fit the free cores
TIME="${TIME:-12:00:00}"    # matches bio-jepa's timeout_min: 720
RESUME="${RESUME:-}"        # checkpoint dir to resume from, if any

# 24 cpus/task comes from the full-node 8-GPU layout (8*24 = 192 = whole node).
# On a SHARED node -- someone else holding GPUs is holding their cores too --
# NGPU*CPUS can exceed what is free and the job just sits in PENDING. Lower
# CPUS (e.g. CPUS=16) if squeue shows Resources/Priority and it won't start.

CONFIG=configs/run_configs/icassp_2026/sl_eat_audioset_xeno_inat.yml

PATCHES="--patch training_params.batch_size=${BS}"
if [[ -n "$RESUME" ]]; then
    PATCHES="$PATCHES --patch resume_from_checkpoint=${RESUME}"
fi

LOG_DIR="$HOME/logs/eat_posttrain_$$"
mkdir -p "$LOG_DIR"

cat > "$LOG_DIR/run.sbatch" <<SBATCH
#!/bin/bash
#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NGPU}
#SBATCH --gpus-per-node=${NGPU}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --job-name=eatsl-posttrain
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
# PyTorch 2.6 flipped torch.load's default to weights_only=True. We trust our
# own checkpoints.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# Redirect all temp / scratch writes to /tmp so pymp-* and tmpXX-wandb-* dirs
# don't litter the repo cwd (and don't fill a home quota mid-run).
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
# wandb local cache goes under logs/ instead of the repo root.
export WANDB_DIR="\$HOME/avex/logs/wandb"
mkdir -p "\$WANDB_DIR"

cd \$HOME/avex
uv sync --group project-dev --group gpu

echo "=== EAT post-train: \$SLURM_NTASKS tasks / ${NGPU} GPUs, per-GPU batch ${BS}, global $((BS * NGPU)) ==="
echo "=== 10 epochs: 1-2 head-only (backbone frozen), 3-10 full fine-tune @ 8e-5 ==="

srun uv run avex train \\
    --config ${CONFIG} \\
    ${PATCHES}
SBATCH

echo "Submitting EAT post-train: ${NGPU} GPUs, per-GPU batch ${BS}, global $((BS * NGPU)), walltime ${TIME}"
[[ -n "$RESUME" ]] && echo "Resuming from: $RESUME"
JOBID=$(sbatch --parsable "$LOG_DIR/run.sbatch")
echo "Job $JOBID submitted."
echo "  out: $LOG_DIR/slurm-${JOBID}.out"
echo "  err: $LOG_DIR/slurm-${JOBID}.err"
echo "Watch with: tail -f $LOG_DIR/slurm-${JOBID}.out"
echo
echo "Check early: 'world size ${NGPU}' (NOT 1), and that epoch 3 logs"
echo "'[Two-stage] Freezing period finished' -- that is the unfreeze + LR reset."
echo "Checkpoints save every epoch under runs/icassp2026/sl_eat_audioset_xeno_inat/<timestamp>/."
