#!/usr/bin/env bash
# Supervised post-training of OUR pre-trained PuPuJEPA-Base (pupujepa_bigrun,
# 200k SSL steps) on AudioSet + Xeno-canto + iNaturalist. Plan step 2.
#
# LAUNCHER, not a batch script -- run it directly on the login node:
#     bash jobs/icassp_2026/train_pupujepa_posttrain.sh              # 8 GPUs, per-GPU 32 -> global 256
#     NGPU=6 BS=32 bash jobs/icassp_2026/train_pupujepa_posttrain.sh # 6 GPUs -> global 192
#     TIME=24:00:00 bash jobs/icassp_2026/train_pupujepa_posttrain.sh
#     RESUME=runs/icassp2026/sl_pupujepa_audioset_xeno_inat/<timestamp> bash ...
# It generates the sbatch file (so NGPU can vary -- #SBATCH directives are static).
#
# DDP: avex only creates a process group when world_size > 1, and world_size comes
# from SLURM_NTASKS -- hence ONE TASK PER GPU. If the log says "world size 1" it is
# silently training on a single card.
#
# BS is PER-GPU (DataLoader + DistributedSampler), so global = BS * NGPU. The
# 8 x 32 default gives the paper's global batch of 256 for the sl-EAT-all recipe.
#
# Requires the two uncommitted avex fixes (get_model.py registry fallback and
# pupujepa_model.py return_features_only=False) -- without them PuPuJEPA cannot be
# trained at all. Verified locally: get_model resolves PupuJEPAModel, builds
# Linear(768 -> 14348), and forwards 32kHz x 10s to (B, 14348).
#
# No --qos and no --exclude: none of the working ESP jobs set a qos, and excluding
# nodes just makes the job queue behind idle hardware.

set -euo pipefail

NGPU="${NGPU:-8}"           # h100-80 nodes have 8 GPUs
BS="${BS:-32}"              # per-GPU batch; global = BS * NGPU (8x32 = 256 = paper)
CPUS="${CPUS:-24}"          # cpus PER TASK; NGPU*CPUS must fit the free cores
TIME="${TIME:-24:00:00}"
RESUME="${RESUME:-}"

CONFIG=configs/run_configs/icassp_2026/sl_pupujepa_audioset_xeno_inat.yml

PATCHES="--patch training_params.batch_size=${BS}"
if [[ -n "$RESUME" ]]; then
    PATCHES="$PATCHES --patch resume_from_checkpoint=${RESUME}"
fi

LOG_DIR="$HOME/logs/pupujepa_posttrain_$$"
mkdir -p "$LOG_DIR"

cat > "$LOG_DIR/run.sbatch" <<SBATCH
#!/bin/bash
#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NGPU}
#SBATCH --gpus-per-node=${NGPU}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --job-name=pupu-posttrain
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
export WANDB_DIR="\$HOME/avex/logs/wandb"
mkdir -p "\$WANDB_DIR"

cd \$HOME/avex
uv sync --group project-dev --group gpu

echo "=== PuPuJEPA post-train: \$SLURM_NTASKS tasks / ${NGPU} GPUs, per-GPU ${BS}, global $((BS * NGPU)) ==="
echo "=== 10 epochs: 1-2 head-only (backbone frozen) @1e-4, 3-10 full @8e-5 ==="

srun uv run avex train \\
    --config ${CONFIG} \\
    ${PATCHES}
SBATCH

echo "Submitting PuPuJEPA post-train: ${NGPU} GPUs, per-GPU ${BS}, global $((BS * NGPU)), walltime ${TIME}"
[[ -n "$RESUME" ]] && echo "Resuming from: $RESUME"
JOBID=$(sbatch --parsable "$LOG_DIR/run.sbatch")
echo "Job $JOBID submitted."
echo "  out: $LOG_DIR/slurm-${JOBID}.out"
echo "  err: $LOG_DIR/slurm-${JOBID}.err"
echo "Watch: tail -f $LOG_DIR/slurm-${JOBID}.err"
echo
echo "Check early, in order:"
echo "  1. 'Set CUDA device N' x ${NGPU}   (absent => CUDA broken, see the cu128 pin)"
echo "  2. 'world size ${NGPU}'  NOT 1     (1 => DDP did not engage)"
echo "  3. PuPuJEPA checkpoint loads without a hard-fail on unmatched tensors"
echo "  4. first-epoch it/s -- if far below the EAT run's ~1 it/s, AudioSet is"
echo "     resampling on the fly at 32kHz (see the data config header)"
