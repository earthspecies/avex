"""Export the PupuJEPA backbone from an avex training checkpoint to safetensors.

The PupuJEPA wrapper loads its weights with safetensors.load_file from
extra_config.checkpoint_path, and hard-fails if any deployed-path tensor
(teacher / patch_embed / rope_encoder) is missing. avex training checkpoints are
.pt files holding {"model_state_dict": {"backbone.*": ..., "classifier.*": ...}}.
This keeps the backbone.* tensors, strips the prefix, and writes them out -- the
same file the old run's backbone_final.safetensors was.

Usage:
    uv run --group project-dev python scripts/export_pupujepa_backbone.py \
        --ckpt runs/icassp2026/<run>/<timestamp>/final_model.pt \
        --out  runs/icassp2026/<run>/backbone_final.safetensors \
        [--init <SSL model.safetensors>]   # also report how many deployed tensors changed
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

DEPLOYED = ("teacher.", "patch_embed.", "rope_encoder.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path, default=None)
    p.add_argument("--all-deployed", action="store_true",
                   help="treat every backbone tensor as deployed (models without training-only "
                        "parts, e.g. the XC_1M MAE); default: PupuJEPA's teacher/patch_embed/rope_encoder")
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model_state_dict", raw)
    bb = {k[len("backbone."):]: v.contiguous() for k, v in sd.items() if k.startswith("backbone.")}
    deployed = list(bb) if args.all_deployed else [k for k in bb if k.startswith(DEPLOYED)]
    if not deployed:
        raise SystemExit(f"no deployed-path tensors under backbone.* in {args.ckpt}")
    print(f"{args.ckpt}: {len(bb)} backbone tensors ({len(deployed)} deployed-path), "
          f"{len(sd) - len(bb)} non-backbone dropped: {[k for k in sd if not k.startswith('backbone.')]}")

    if args.init is not None:
        init = load_file(args.init)
        missing = [k for k in deployed if k not in init]
        if missing:
            raise SystemExit(f"deployed tensors absent from init: {missing[:5]}")
        changed = sum(not torch.equal(bb[k].float(), init[k].float()) for k in deployed)
        print(f"deployed-path tensors changed vs init: {changed}/{len(deployed)}")
        if changed == 0:
            raise SystemExit("backbone is bit-identical to the SSL init -- post-training did not update it")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(bb, str(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
