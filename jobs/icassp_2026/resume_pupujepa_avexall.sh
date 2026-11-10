#!/usr/bin/env bash
# RESUME the PupuJEPA avex-all post-train (job 91713 hit its 24 h limit during
# epoch 9) from checkpoint_epoch_008.pt, then export + eval -- LAUNCHER, run on the
# login node:
#     bash jobs/icassp_2026/resume_pupujepa_avexall.sh                  # 8 GPUs x 32
#     NGPU=5 BS=51 bash jobs/icassp_2026/resume_pupujepa_avexall.sh     # 5 GPUs x 51 = 255
#
#   B' resume      8xH100   epochs 9-10 (~6.5 h), new timestamped run dir
#   C  export      cpu      backbone -> backbone_final.safetensors, afterok:B'
#   D  eval post   2xH100   eval_pupujepa_table6.sh MODEL=avexall, afterok:C
#
# Needs the resume fix in avex/training/train.py (_load_checkpoint): before it,
# resuming past the freeze period rebuilt stage 2 from scratch at the first
# resumed epoch (AdamW moments reset, LR warm-up restarted) and only rank 0
# loaded the checkpoint. Verified on this exact checkpoint: resumes at epoch 9,
# scheduler step 72228 (= 6 x 12038), lr 1.406e-05 (the log shows 1.41e-05 at
# the end of epoch 8).
#
# Check in the .err log: "[Resume] start epoch 9 | scheduler step 72228 | lr 1.406e-05"
# on all 8 ranks, and NO "[Two-stage] Freezing period finished at epoch 9".
#
# Cancel the stale chain from overnight_pupujepa.sh first:  scancel 91714 91715

set -euo pipefail

CONFIG=configs/run_configs/icassp_2026/sl_pupujepa_avex_all_32k.yml
CKPT=runs/icassp2026/sl_pupujepa_avex_all_32k/2026-09-28_00-05-50/checkpoint_epoch_008.pt
NGPU="${NGPU:-8}"; BS="${BS:-32}"; CPUS="${CPUS:-24}"; TIME="${TIME:-12:00:00}"
# Keep the GLOBAL batch at ~256. The restored scheduler keeps the original run's
# total step count, so steps/epoch must stay ~12,038: with 5 x 32 (global 160,
# 19,262 steps/epoch) the cosine bottoms out mid-epoch 9 and climbs back up
# (lr 5.9e-06 at the end instead of 1e-06). 5 x 51 = 255 gives 12,086 steps/epoch
# and lr 4.387e-06 / 1.000e-06 at the end of epochs 9 / 10 vs 4.414e-06 / 1.000e-06
# for the original 8 x 32 (checked on the real checkpoint).
GLOBAL=$((NGPU * BS))
if (( GLOBAL < 251 || GLOBAL > 261 )); then
    echo "ERROR: NGPU x BS = $GLOBAL; keep it within 2% of 256 (e.g. NGPU=5 BS=51, NGPU=4 BS=64, NGPU=6 BS=43)" >&2
    exit 1
fi

cd ~/avex
[[ -f "$CKPT" ]] || { echo "ERROR: $CKPT not found" >&2; exit 1; }
if ls runs/icassp2026/sl_pupujepa_avex_all_32k/*/final_model.pt >/dev/null 2>&1; then
    echo "ERROR: a final_model.pt already exists under runs/icassp2026/sl_pupujepa_avex_all_32k -- nothing to resume?" >&2
    exit 1
fi

LOG_DIR="$HOME/logs/pupujepa_avexall_resume_$$"
mkdir -p "$LOG_DIR"

cat > "$LOG_DIR/train.sbatch" <<SBATCH
#!/bin/bash
#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NGPU}
#SBATCH --gpus-per-node=${NGPU}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --job-name=pa-resume
#SBATCH --output=$LOG_DIR/train-%j.out
#SBATCH --error=$LOG_DIR/train-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$LOG_DIR/nccl_trace_rank_
export NCCL_DEBUG=WARN
export AVEX_NCCL_TIMEOUT_MINUTES=30
export WANDB_DIR="\$HOME/avex/logs/wandb"
mkdir -p "\$WANDB_DIR"
cd \$HOME/avex
uv sync --group project-dev --group gpu
echo "=== PupuJEPA avex-all RESUME from ${CKPT}: epochs 9-10, ${NGPU} GPUs x ${BS} = ${GLOBAL} ==="
srun uv run avex train --config ${CONFIG} \\
    --patch training_params.batch_size=${BS} \\
    --patch resume_from_checkpoint=${CKPT}
SBATCH

B=$(sbatch --parsable "$LOG_DIR/train.sbatch")
C=$(sbatch --parsable --dependency=afterok:$B jobs/icassp_2026/export_pupujepa_avexall.sh)
D=$(sbatch --parsable --dependency=afterok:$C --export=ALL,MODEL=avexall --job-name=t6-avexall jobs/icassp_2026/eval_pupujepa_table6.sh)

cat <<MSG
Submitted:
  B' resume     $B   $LOG_DIR/train-$B.err
  C  export     $C   ~/logs/pupu-export-$C.log       (after B')
  D  eval post  $D   ~/logs/t6-avexall-${D}_{1,2}.log (after C)
MSG
