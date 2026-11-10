#!/usr/bin/env bash
# After the sl_pupujepa_avex_all_32k post-train finishes: export its final
# backbone to runs/icassp2026/sl_pupujepa_avex_all_32k/backbone_final.safetensors,
# the fixed path the pupujepa_avexall*.yml eval run configs point at.
# Refuses to run if there is not exactly one finished run (final_model.pt), and
# fails if the backbone is bit-identical to the SSL init.
#
# Submit:  sbatch jobs/icassp_2026/export_pupujepa_avexall.sh
# (overnight_pupujepa.sh submits it after training.)

#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output="/home/%u/logs/pupu-export-%j.log"
#SBATCH --job-name="pa-export"

set -euo pipefail
cd ~/avex
uv sync --group project-dev

ROOT=runs/icassp2026/sl_pupujepa_avex_all_32k
mapfile -t FINALS < <(ls -1 "$ROOT"/*/final_model.pt 2>/dev/null || true)
if [[ ${#FINALS[@]} -ne 1 ]]; then
    echo "ERROR: expected exactly one $ROOT/<timestamp>/final_model.pt, found ${#FINALS[@]}: ${FINALS[*]:-none}" >&2
    exit 1
fi
echo "exporting ${FINALS[0]}"
uv run --group project-dev python scripts/export_pupujepa_backbone.py \
    --ckpt "${FINALS[0]}" \
    --out "$ROOT/backbone_final.safetensors" \
    --init "$HOME/bio-jepa/checkpoints/pupujepa_bigrun/step-0200000_loss-0.161731/model.safetensors"
