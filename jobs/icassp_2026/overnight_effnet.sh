#!/usr/bin/env bash
# EfficientNet-B0 32 kHz (Marius's effnet_32khz_v0): post-training + Tables 6-9 -- LAUNCHER,
# run on the login node:
#     bash jobs/icassp_2026/overnight_effnet.sh
#
# Copy of overnight_xenomae.sh: same trainer, recipe, data and NCCL/uv settings. The config
# sl_effnet32k_avex_all.yml differs from sl_xenomae_avex_all_32k.yml only in model_spec and
# init_checkpoint (Marius's encoder, loaded strictly; his 8,463-class head discarded).
#
#   B  post-train   8xH100   10 epochs                                        starts when 8 GPUs free
#   C  copy         cpu      final_model.pt -> runs/icassp2026/sl_effnet32k_avex_all/   afterok:B
#   D  eval post    4xH100   Tables 6-9, MODEL=avexall                        afterok:C
#   E  eval SSL     4xH100   Tables 6-9, MODEL=ssl                            after:B (once B STARTED)
#
# Check in B's .err: "world size 8", "init_checkpoint gs://.../effnet_32khz_v0/final_model.pt:
# 358/360 ...", and at epoch 3 "[Two-stage] Unfroze ..." + "[Two-stage] Rebuilt DDP".

set -euo pipefail

CONFIG=configs/run_configs/icassp_2026/sl_effnet32k_avex_all.yml
NGPU=8; BS=32; CPUS=24; TIME=24:00:00
ROOT=runs/icassp2026/sl_effnet32k_avex_all

cd ~/avex
if [[ -e "$ROOT" ]]; then
    echo "ERROR: $ROOT already exists -- the export step needs exactly one run there. Move it aside." >&2
    exit 1
fi
LOG_DIR="$HOME/logs/effnet32k_avexall_$$"
mkdir -p "$LOG_DIR"

cat > "$LOG_DIR/train.sbatch" <<SBATCH
#!/bin/bash
#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NGPU}
#SBATCH --gpus-per-node=${NGPU}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --job-name=eff-train
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
# The cluster points uv at the node-local /scratch, which is full on several nodes
# (jobs 94459/94462: "No space left on device" during uv sync). Use home instead:
# shared wheel cache, and a per-job environment removed when the job ends.
export UV_CACHE_DIR="\$HOME/cache/uv"
export UV_PROJECT_ENVIRONMENT="\$HOME/cache/jobenvs/avex-\${SLURM_JOB_ID}"
trap 'rm -rf "\$UV_PROJECT_ENVIRONMENT"' EXIT
mkdir -p "\$WANDB_DIR"
cd \$HOME/avex
uv sync --group project-dev --group gpu
echo "=== EfficientNet-B0 32k avex-all post-train: \$SLURM_NTASKS tasks, per-GPU ${BS}, global $((BS * NGPU)) ==="
# Run the env's own binary: inside srun the cluster resets UV_* to the full /scratch, so
# 'srun uv run' re-installed there and failed on every rank (job 94467).
srun \$UV_PROJECT_ENVIRONMENT/bin/avex train --config ${CONFIG} --patch training_params.batch_size=${BS}
SBATCH

cat > "$LOG_DIR/export.sbatch" <<SBATCH
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=eff-copy
#SBATCH --output=$LOG_DIR/export-%j.log
set -euo pipefail
cd \$HOME/avex
mapfile -t FINALS < <(ls -1 ${ROOT}/*/final_model.pt 2>/dev/null || true)
if [[ \${#FINALS[@]} -ne 1 ]]; then echo "ERROR: expected one final_model.pt, found \${#FINALS[@]}" >&2; exit 1; fi
cp -n "\${FINALS[0]}" ${ROOT}/final_model.pt
ls -la ${ROOT}/final_model.pt
SBATCH

B=$(sbatch --parsable "$LOG_DIR/train.sbatch")
C=$(sbatch --parsable --dependency=afterok:$B "$LOG_DIR/export.sbatch")
D=$(sbatch --parsable --dependency=afterok:$C --export=ALL,MODEL=avexall --job-name=eff-post jobs/icassp_2026/eval_effnet_tables.sh)
E=$(sbatch --parsable --dependency=after:$B --export=ALL,MODEL=ssl --job-name=eff-ssl jobs/icassp_2026/eval_effnet_tables.sh)

cat <<MSG
Submitted:
  B post-train  $B   $LOG_DIR/train-$B.err
  C copy        $C   $LOG_DIR/export-$C.log              (after B succeeds)
  D eval post   $D   ~/logs/eff-post-${D}_{1..4}.log     (after C)
  E eval SSL    $E   ~/logs/eff-ssl-${E}_{1..4}.log      (once B has started)
MSG
