#!/usr/bin/env bash
# REPRODUCE the AVEX paper's EffNetB0-all: EfficientNet-B0 from ImageNet,
# post-trained on AudioSet + Xeno-canto + iNat + Watkins + Animal Sound Archive
# at 16 kHz (paper Tables 3/4/5). Reference: BEANS Classification probe 0.800.
#
# LAUNCHER, not a batch script -- run it on the login node:
#     bash jobs/icassp_2026/train_effnetb0_all_repro.sh              # 50 epochs (paper), 8 GPUs x 32 = 256
#     EPOCHS=10 bash jobs/icassp_2026/train_effnetb0_all_repro.sh    # shorter; cosine schedule spans 10
#     NGPU=4 bash jobs/icassp_2026/train_effnetb0_all_repro.sh       # 128 global
#     RESUME=runs/icassp2026/sl_effnetb0_all_repro/<timestamp>/checkpoint_epoch_00N.pt bash ...
#
# EPOCHS < 50 is NOT a faithful reproduction: the cosine schedule and total steps
# differ from the paper, so the result is "how this recipe does in N epochs", not
# a check against Table 3.
#
# Checks in the .err log: "world size N" (not 1); loss falling; no init_checkpoint
# line (ImageNet init comes from torchvision) and no [Two-stage] lines (no freeze).
# No --qos and no --exclude.

set -euo pipefail

NGPU="${NGPU:-8}"           # h100-80 nodes have 8 GPUs
BS="${BS:-32}"              # per-GPU batch; global = BS * NGPU (8x32 = 256 = paper)
CPUS="${CPUS:-24}"          # cpus PER TASK; NGPU*CPUS must fit the free cores
TIME="${TIME:-24:00:00}"
RESUME="${RESUME:-}"
EPOCHS="${EPOCHS:-50}"       # paper = 50; e.g. EPOCHS=10 for a run that finishes in ~half a day

CONFIG=configs/run_configs/icassp_2026/sl_effnetb0_all_repro.yml

PATCHES="--patch training_params.batch_size=${BS} --patch training_params.train_epochs=${EPOCHS}"
if [[ -n "$RESUME" ]]; then
    PATCHES="$PATCHES --patch resume_from_checkpoint=${RESUME}"
fi

LOG_DIR="$HOME/logs/sl_effnetb0_all_repro_$$"
mkdir -p "$LOG_DIR"

cat > "$LOG_DIR/run.sbatch" <<SBATCH
#!/bin/bash
#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NGPU}
#SBATCH --gpus-per-node=${NGPU}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --job-name=effnet-posttrain
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
# NCCL flight recorder: if ranks desync again (job 91246 hung at epoch 2 step 378
# with rank 2 four collectives ahead), dump each rank's recent collectives and
# their call sites so we can see WHICH operation diverged. Logging only -- no
# effect on training.
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$LOG_DIR/nccl_trace_rank_
export NCCL_DEBUG=WARN
export WANDB_DIR="\$HOME/avex/logs/wandb"
mkdir -p "\$WANDB_DIR"

cd \$HOME/avex
uv sync --group project-dev --group gpu

echo "=== EffNetB0-all REPRO post-train: \$SLURM_NTASKS tasks / ${NGPU} GPUs, per-GPU ${BS}, global $((BS * NGPU)) ==="
echo "=== ${EPOCHS} epochs, full model from ImageNet init, lr 5e-4, no freeze ==="

srun uv run avex train \\
    --config ${CONFIG} \\
    ${PATCHES}
SBATCH

echo "Submitting EffNetB0-all REPRO post-train: ${EPOCHS} epochs, ${NGPU} GPUs, per-GPU ${BS}, global $((BS * NGPU)), walltime ${TIME}"
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
