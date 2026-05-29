#!/usr/bin/env bash
# Sync repo to Slurm login node, honoring .gitignore (no per-folder --exclude flags needed).
#
# Usage:
#   ./jobs/sync_to_slurm.sh           # sync
#   ./jobs/sync_to_slurm.sh --dry-run # preview only
#
# Override host or destination:
#   REMOTE_HOST=slurm REMOTE_DIR=/path/to/dir ./jobs/sync_to_slurm.sh

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-slurm}"
REMOTE_DIR="${REMOTE_DIR:-/home/marius_miron_earthspecies_org/code/representation_learning}"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DRY_RUN=()
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=(-n)
    shift
fi

cd "$SRC_DIR"

echo "Source:      ${SRC_DIR}/"
echo "Destination: ${REMOTE_HOST}:${REMOTE_DIR}/"
echo "Excludes:    .gitignore rules (dir-merge filter) + .git/"
[[ ${#DRY_RUN[@]} -gt 0 ]] && echo "Mode:        dry-run"
echo

rsync -avz --progress "${DRY_RUN[@]}" \
    --include='configs/run_configs/pretrained/***' \
    --filter='dir-merge,- .gitignore' \
    --exclude='.git/' \
    -e ssh \
    "${SRC_DIR}/" \
    "${REMOTE_HOST}:${REMOTE_DIR}/"

echo
echo "Done."
