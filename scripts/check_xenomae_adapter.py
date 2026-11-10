"""Checks for the XC_1M MAE avex adapter (avex/models/xenomae_model.py). CPU-only.

1. registry: model name "xenomae" resolves to XenoMAEModel
2. preprocessing parity: adapter.preprocess == the MVP's preprocess_audio, per window
3. embedding parity: adapter output == MVP model on MVP-preprocessed windows, averaged
4. both poolings, short/long clips, 16 kHz input
5. post-training path: classifier logits, gradients reach every backbone tensor

Usage:
    uv run --group project-dev python scripts/check_xenomae_adapter.py \
        --weights /mnt/home/moritz_mvp/model.safetensors
"""

import argparse

import torch

import avex.run_train  # noqa: F401  (model registration)
from avex.configs import AudioConfig
from avex.models.utils.registry import get_model_class
from avex.models.xenomae import xenomae as mvp
from avex.models.xenomae_model import XenoMAEModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    check("registry: 'xenomae' -> XenoMAEModel", get_model_class("xenomae") is XenoMAEModel)

    ac = AudioConfig(sample_rate=32000, representation="raw", normalize=False,
                     target_length_seconds=10, window_selection="random")
    extra = {"checkpoint_path": args.weights}
    m = {pl: XenoMAEModel(device="cpu", audio_config=ac, return_features_only=True,
                          extra_config={**extra, "pooling": pl}).eval() for pl in ("mean", "cls")}
    ref = mvp.load_model(args.weights)

    # 12.3 s clip -> windows 0-5, 5-10, and a 2.3 s tail (> 1.25 s, kept)
    clip = torch.randn(int(12.3 * 32000)) * torch.linspace(0.05, 0.5, int(12.3 * 32000))
    wins = m["mean"]._windows(clip)
    check("windowing: 12.3 s -> 3 windows of 5 s", tuple(wins.shape) == (3, 160000), str(tuple(wins.shape)))

    raw_windows = [clip[:160000], clip[160000:320000], clip[320000:]]
    ref_spec = torch.cat([mvp.preprocess_audio(w, 32000) for w in raw_windows])
    ad_spec = m["mean"].preprocess(wins)
    d = (ref_spec - ad_spec).abs().max().item()
    check("preprocessing parity vs MVP preprocess_audio", d < 1e-4, f"(max abs diff {d:.2e})")

    with torch.inference_mode():
        tok = ref(ref_spec, return_tokens=True)
        for pl, want in (("mean", tok[:, 1:].mean(1).mean(0)), ("cls", tok[:, 0].mean(0))):
            got = m[pl](clip.unsqueeze(0))
            check(f"embedding parity ({pl})", got.shape == (1, 1, 768) and torch.allclose(got[0, 0], want, atol=1e-4),
                  f"(shape {tuple(got.shape)}, max diff {(got[0, 0] - want).abs().max().item():.2e})")
        check("README default (CLS of a single 5 s clip) reproduced",
              torch.allclose(m["cls"](raw_windows[0].unsqueeze(0))[0, 0], ref(ref_spec[:1])[0], atol=1e-4))
        short = m["mean"](torch.randn(2, 32000))
        check("short 1 s clips -> (2, 1, 768)", tuple(short.shape) == (2, 1, 768))

    ac16 = AudioConfig(sample_rate=16000, representation="raw", normalize=False,
                       target_length_seconds=10, window_selection="random")
    m16 = XenoMAEModel(device="cpu", audio_config=ac16, return_features_only=True, extra_config=extra).eval()
    with torch.inference_mode():
        out16 = m16(torch.randn(2, 16000 * 10))
    check("16 kHz input resampled -> (2, 1, 768)", tuple(out16.shape) == (2, 1, 768))

    head = XenoMAEModel(device="cpu", audio_config=ac, num_classes=14747, extra_config=extra).train()
    logits = head(torch.randn(2, 32000 * 10))
    check("classifier logits (2, 14747)", tuple(logits.shape) == (2, 14747))
    logits.sum().backward()
    missing = [n for n, q in head.backbone.named_parameters() if q.grad is None or q.grad.abs().sum() == 0]
    n_bb = sum(1 for _ in head.backbone.parameters())
    check("gradients reach every backbone tensor", not missing, f"({n_bb - len(missing)}/{n_bb})")
    check("exposes .backbone for two-stage freeze/unfreeze", hasattr(head, "backbone"))

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
