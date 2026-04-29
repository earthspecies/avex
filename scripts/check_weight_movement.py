"""Confirm CLAP audio encoder, text encoder, and projection layers all moved
during training, by comparing checkpoint weights against a fresh init built
with the same config.

For each component, prints:
    - mean abs delta (scaled by mean abs of init weights)
    - fraction of params with delta > 0 (i.e. that actually moved)
    - count of frozen params (delta == 0 exactly)
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Iterable

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


COMPONENTS = {
    "audio_encoder": ("audio_encoder.",),
    "text_encoder": ("text_encoder.",),
    "audio_projection": ("audio_projection.",),
    "text_projection": ("text_projection.",),
    "logit_scale": ("logit_scale",),
}


def classify(name: str) -> str | None:
    for label, prefixes in COMPONENTS.items():
        for p in prefixes:
            if name.startswith(p) or name == p:
                return label
    return None


def build_fresh(config_path: str, device: torch.device):
    from avex.configs import RunConfig
    from avex.models.get_model import get_model

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if "model_spec" in raw and isinstance(raw["model_spec"], dict):
        raw["model_spec"]["device"] = "cuda" if device.type == "cuda" else "cpu"
    cfg = RunConfig(**raw)
    cfg.model_spec.device = "cuda" if device.type == "cuda" else "cpu"
    model = get_model(cfg.model_spec, num_classes=0).to(device)
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="/mnt/home/avex/runs/clap/clap_chain_synthetic_xc_inat/"
                           "2026-04-28_06-55-37/best_model.pt")
    p.add_argument("--config",
                   default="/mnt/home/avex/runs/clap/clap_chain_synthetic_xc_inat/"
                           "2026-04-28_06-55-37/config.yml")
    args = p.parse_args()

    device = torch.device("cpu")  # weight comparison doesn't need GPU
    logger.info("Building fresh model from config (with audio_encoder_init_from in effect)...")
    fresh = build_fresh(args.config, device)
    fresh_state = fresh.state_dict()

    logger.info("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    trained_state = ckpt["model_state_dict"]
    logger.info("Checkpoint epoch: %s", ckpt.get("epoch"))

    # For each parameter present in both, compute delta
    component_stats: dict[str, dict] = defaultdict(lambda: {
        "n_tensors": 0, "n_params": 0, "moved_params": 0,
        "sum_abs_delta": 0.0, "sum_abs_init": 0.0, "max_delta": 0.0,
    })

    missing = [k for k in fresh_state if k not in trained_state]
    extra = [k for k in trained_state if k not in fresh_state]
    if missing:
        logger.warning("Fresh has %d keys missing in checkpoint (e.g. %s)",
                       len(missing), missing[:3])
    if extra:
        logger.warning("Checkpoint has %d keys missing in fresh (e.g. %s)",
                       len(extra), extra[:3])

    for name, init_w in fresh_state.items():
        if name not in trained_state:
            continue
        trained_w = trained_state[name]
        if init_w.shape != trained_w.shape:
            logger.warning("Shape mismatch on %s: %s vs %s", name, init_w.shape,
                           trained_w.shape)
            continue
        comp = classify(name)
        if comp is None:
            continue
        init_w = init_w.float()
        trained_w = trained_w.float()
        delta = (trained_w - init_w).abs()
        s = component_stats[comp]
        s["n_tensors"] += 1
        s["n_params"] += init_w.numel()
        s["moved_params"] += int((delta > 0).sum().item())
        s["sum_abs_delta"] += float(delta.sum().item())
        s["sum_abs_init"] += float(init_w.abs().sum().item())
        s["max_delta"] = max(s["max_delta"], float(delta.max().item()))

    print()
    print("=" * 90)
    print(f"{'Component':<20} {'tensors':>8} {'params':>12} "
          f"{'moved %':>10} {'mean|Δ|':>12} {'rel|Δ|':>10} {'max|Δ|':>10}")
    print("=" * 90)
    for comp in ("audio_encoder", "text_encoder",
                 "audio_projection", "text_projection", "logit_scale"):
        s = component_stats.get(comp)
        if not s or s["n_params"] == 0:
            print(f"{comp:<20} (none found)")
            continue
        moved_pct = 100.0 * s["moved_params"] / s["n_params"]
        mean_abs = s["sum_abs_delta"] / s["n_params"]
        rel = s["sum_abs_delta"] / max(s["sum_abs_init"], 1e-12)
        print(f"{comp:<20} {s['n_tensors']:>8d} {s['n_params']:>12d} "
              f"{moved_pct:>9.2f}% {mean_abs:>12.6f} {rel:>9.4f} "
              f"{s['max_delta']:>10.4f}")
    print("=" * 90)
    print()
    print("moved %    : fraction of scalar params that differ from init (>0 abs delta)")
    print("mean|Δ|    : average absolute change per scalar")
    print("rel|Δ|     : sum|Δ| / sum|init|  — proportional movement")
    print("max|Δ|     : largest single-scalar change")


if __name__ == "__main__":
    main()
