#!/usr/bin/env bash
# Overnight PupuJEPA chain -- LAUNCHER, run on the login node:
#     bash jobs/icassp_2026/overnight_pupujepa.sh
#
# Submits, with SLURM dependencies (each step starts only if the previous succeeded):
#
#   A  preflight   cpu      data/label check for the new 32 kHz 5-source mix   ~minutes
#   B  post-train  8xH100   sl_pupujepa_avex_all_32k.yml, afterok:A            ~12-14 h
#   C  export      cpu      backbone -> backbone_final.safetensors, afterok:B  ~minutes
#   D  eval post   2xH100   eval_pupujepa_table6.sh MODEL=avexall, afterok:C
#   E  eval SSL    2xH100   eval_pupujepa_table6.sh MODEL=bigrun, NO dependency (starts now)
#
# B is the recipe that reproduced the paper's sl-EAT-all (sl_eat_all_repro.yml),
# on the fixed trainer (rebuilds DDP after the epoch-3 unfreeze -- the old
# 2026-08-31 PupuJEPA run never averaged backbone gradients across GPUs).
#
# Check in B's .err log, in order:
#   1. "world size 8"
#   2. "Loaded PupuJEPA checkpoint .../step-0200000..."
#   3. "PupuJEPA embedding layer: final"
#   4. epoch 3: "[Two-stage] Unfroze N backbone parameter tensors" AND "[Two-stage] Rebuilt DDP"
#
# No --qos and no --exclude.

set -euo pipefail

CONFIG=configs/run_configs/icassp_2026/sl_pupujepa_avex_all_32k.yml
NGPU=8; BS=32; CPUS=24; TIME=24:00:00

cd ~/avex
if [[ -e runs/icassp2026/sl_pupujepa_avex_all_32k ]]; then
    echo "ERROR: runs/icassp2026/sl_pupujepa_avex_all_32k already exists -- the export step" >&2
    echo "       needs exactly one run there. Move it aside first." >&2
    exit 1
fi

LOG_DIR="$HOME/logs/pupujepa_avexall_$$"
mkdir -p "$LOG_DIR"

# ---- A: preflight -----------------------------------------------------------
cat > "$LOG_DIR/preflight.sbatch" <<SBATCH
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=pa-preflight
#SBATCH --output=$LOG_DIR/preflight-%j.log
set -euo pipefail
cd \$HOME/avex
uv sync --group project-dev
uv run --group project-dev python scripts/preflight_posttrain_labels.py --config ${CONFIG}
SBATCH

# ---- B: post-train ----------------------------------------------------------
cat > "$LOG_DIR/train.sbatch" <<SBATCH
#!/bin/bash
#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NGPU}
#SBATCH --gpus-per-node=${NGPU}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --job-name=pa-train
#SBATCH --output=$LOG_DIR/train-%j.out
#SBATCH --error=$LOG_DIR/train-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
# Same stall protection as the EAT repro run (job 91402): 30-min NCCL timeout,
# and the flight recorder dumps each rank's collectives if one ever fires.
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$LOG_DIR/nccl_trace_rank_
export NCCL_DEBUG=WARN
export AVEX_NCCL_TIMEOUT_MINUTES=30
export WANDB_DIR="\$HOME/avex/logs/wandb"
mkdir -p "\$WANDB_DIR"
cd \$HOME/avex
uv sync --group project-dev --group gpu
echo "=== PupuJEPA avex-all post-train: \$SLURM_NTASKS tasks, per-GPU ${BS}, global $((BS * NGPU)) ==="
srun uv run avex train --config ${CONFIG} --patch training_params.batch_size=${BS}
SBATCH

A=$(sbatch --parsable "$LOG_DIR/preflight.sbatch")
B=$(sbatch --parsable --dependency=afterok:$A "$LOG_DIR/train.sbatch")
C=$(sbatch --parsable --dependency=afterok:$B jobs/icassp_2026/export_pupujepa_avexall.sh)
D=$(sbatch --parsable --dependency=afterok:$C --export=ALL,MODEL=avexall --job-name=t6-avexall jobs/icassp_2026/eval_pupujepa_table6.sh)
E=$(sbatch --parsable --export=ALL,MODEL=bigrun --job-name=t6-bigrun jobs/icassp_2026/eval_pupujepa_table6.sh)

cat <<MSG
Submitted:
  A preflight   $A   $LOG_DIR/preflight-$A.log        (must end with PREFLIGHT PASSED)
  B post-train  $B   $LOG_DIR/train-$B.err            (starts after A)
  C export      $C   ~/logs/pupu-export-$C.log        (starts after B)
  D eval post   $D   ~/logs/t6-avexall-${D}_{1,2}.log   (starts after C)
  E eval SSL    $E   ~/logs/t6-bigrun-${E}_{1,2}.log    (starts now)
If A fails, B/C/D stay pending with reason DependencyNeverSatisfied -- scancel them.
MSG
