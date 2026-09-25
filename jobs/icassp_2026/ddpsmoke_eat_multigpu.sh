#!/usr/bin/env bash
# DDP smoke test for the ICASSP-2026 EAT post-training pipeline.
#
# LAUNCHER, not a batch script -- run it directly on the login node:
#     bash jobs/icassp_2026/ddpsmoke_eat_multigpu.sh            # 8 GPUs, per-GPU batch 16
#     NGPU=4 bash jobs/icassp_2026/ddpsmoke_eat_multigpu.sh     # 4 GPUs
#     NGPU=8 BS=32 bash jobs/icassp_2026/ddpsmoke_eat_multigpu.sh
# It generates the sbatch file (so NGPU can vary -- #SBATCH directives are
# static) and submits it, mirroring bio-jepa/scripts/pretrain/ddp_smoke_test.sh.
#
# Answers two questions before we commit real GPUs to the 10-epoch run:
#   1) Does DDP actually engage? avex only creates a process group when
#      world_size > 1, and world_size comes from SLURM_NTASKS -- so this needs
#      ONE TASK PER GPU. The existing train_eat_audioset_xeno_inat_h100.sh has
#      ntasks-per-node=1 and so runs single-GPU regardless of GPUs allocated.
#   2) Does the per-GPU batch size fit in 80GB with the backbone unfrozen?
#      avex has NO gradient accumulation, so whatever we set must fit for real.
#
# batch_size is PER-GPU: it goes straight to the DataLoader alongside a
# DistributedSampler (avex/data/dataset.py:547,614). Global = BS * NGPU.
# The default BS=16 on 8 GPUs gives global 128, the effective batch that the
# production config's lr=1e-4 was chosen for.
#
# SLURM layout (partition, tasks-per-GPU, cpus_per_task) follows the working
# bio-jepa/configs/slurm/h100_80_x8.yaml. Note there is deliberately NO --qos:
# none of the working ESP jobs set one, and `--qos=naturelm` (copied into the
# older avex job scripts) is rejected as an invalid qos specification.

set -euo pipefail

NGPU="${NGPU:-8}"    # h100-80 nodes have 8 GPUs
BS="${BS:-16}"       # per-GPU batch size; global = BS * NGPU

LOG_DIR="$HOME/logs/ddpsmoke_eat_$$"
mkdir -p "$LOG_DIR"

cat > "$LOG_DIR/run.sbatch" <<SBATCH
#!/bin/bash
#SBATCH --partition=h100-80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NGPU}
#SBATCH --gpus-per-node=${NGPU}
#SBATCH --cpus-per-task=24
#SBATCH --time=00:40:00
#SBATCH --job-name=ddpsmoke-eat
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp

cd \$HOME/avex
uv sync --group project-dev --group gpu

echo "=== DDP smoke: \$SLURM_NTASKS tasks / ${NGPU} GPUs, per-GPU batch ${BS}, global $((BS * NGPU)) ==="

srun uv run avex train \\
    --config configs/run_configs/icassp_2026/sl_eat_audioset_xeno_inat_smoke.yml \\
    --patch "training_params.batch_size=${BS}"
SBATCH

echo "Submitting DDP smoke: ${NGPU} GPUs, per-GPU batch ${BS}, global batch $((BS * NGPU))"
JOBID=$(sbatch --parsable "$LOG_DIR/run.sbatch")
echo "Job $JOBID submitted."
echo "  out: $LOG_DIR/slurm-${JOBID}.out"
echo "  err: $LOG_DIR/slurm-${JOBID}.err"
echo "Watch with: tail -f $LOG_DIR/slurm-${JOBID}.out"
echo
echo "Success looks like: 'Initializing distributed training with rank N, world size ${NGPU}'"
echo "and 'Wrapping model with DDP on rank N' -- world size must be ${NGPU}, not 1."
