from __future__ import annotations

import psutil
import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.eat.audio_model import Model as EATModel


def test_eat_pretrain_forward_cpu() -> None:
    """Smoke-test the EAT pre-training forward path on CPU."""

    # Minimal audio config (mel-spectrogram)
    audio_cfg = AudioConfig(
        sample_rate=2000,
        n_fft=128,
        hop_length=128,
        n_mels=128,
        representation="mel_spectrogram",
        normalize=False,
        target_length_seconds=1,  # 1-s clips simplify runtime
    )

    # ------------------------------------------------------------------
    #  Memory diagnostics BEFORE model instantiation
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"[DEBUG] CUDA is available → current GPU mem: {gpu_mem_mb:.2f} MB")
    else:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024**2
        print(f"[DEBUG] CUDA not available → current RSS: {rss_mb:.2f} MB")

    # Instantiate model in **pretraining** mode (no classification head)
    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EATModel(
        num_classes=1,  # dummy – ignored in pretraining mode
        device=model_device,
        audio_config=audio_cfg,
        pretraining_mode=True,
        enable_ema=True,
    )

    # ------------------------------------------------------------------
    #  Memory diagnostics AFTER model instantiation
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cur = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(
            (
                "[DEBUG] After model creation → "
                f"GPU mem: {cur:.2f} MB / {peak:.2f} MB (max)"
            )
        )
    else:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024**2
        print(f"[DEBUG] After model creation → RSS: {rss_mb:.2f} MB")

    # Print parameter count for reference
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[DEBUG] Model parameters: {params_m:.2f} M")

    model.eval()

    # Dummy batch: 2 random 1-sec clips at 16 kHz
    wav = torch.randn(1, 2000)
    print(wav.shape)

    with torch.no_grad():
        out = model(wav)

    # ------------------------------------------------------------------
    #  Memory diagnostics AFTER forward pass
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        cur = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(
            (
                "[DEBUG] After forward pass → "
                f"GPU mem: {cur:.2f} MB / {peak:.2f} MB (max)"
            )
        )
    else:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024**2
        print(f"[DEBUG] After forward pass → RSS: {rss_mb:.2f} MB")

    # Assertions --------------------------------------------------------
    assert isinstance(out, dict), "EAT pretraining forward must return a dict"
    assert "losses" in out, "Output dict must contain 'losses' key"
    assert out["losses"], "Losses dict cannot be empty"
    for k, v in out["losses"].items():
        assert v.dim() == 0, f"Loss '{k}' should be a scalar tensor"


if __name__ == "__main__":
    print("Running test_eat_pretrain_forward_cpu")
    test_eat_pretrain_forward_cpu()
