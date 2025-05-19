import torch
import psutil

from representation_learning.models.eat.audio_model import Model as EATModel
from representation_learning.configs import AudioConfig


def test_eat_pretrain_forward_cpu():
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
        print(f"[DEBUG] CUDA is available → current GPU mem: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    else:
        process = psutil.Process()
        print(f"[DEBUG] CUDA not available → current RSS: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    # Instantiate model in **pretraining** mode (no classification head)
    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EATModel(
        num_classes=1,               # dummy – ignored in pretraining mode
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
        print(
            f"[DEBUG] After model creation → GPU mem: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB / "
            f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB (max)"
        )
    else:
        process = psutil.Process()
        print(f"[DEBUG] After model creation → RSS: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    # Print parameter count for reference
    print(
        f"[DEBUG] Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M"
    )

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
        print(
            f"[DEBUG] After forward pass → GPU mem: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB / "
            f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB (max)"
        )
    else:
        process = psutil.Process()
        print(f"[DEBUG] After forward pass → RSS: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    # Assertions --------------------------------------------------------
    assert isinstance(out, dict), "EAT pretraining forward must return a dict"
    assert "losses" in out, "Output dict must contain 'losses' key"
    assert out["losses"], "Losses dict cannot be empty"
    for k, v in out["losses"].items():
        assert v.dim() == 0, f"Loss '{k}' should be a scalar tensor"

if __name__ == "__main__":
    print("Running test_eat_pretrain_forward_cpu")
    test_eat_pretrain_forward_cpu()
