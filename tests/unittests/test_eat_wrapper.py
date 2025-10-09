from __future__ import annotations

import torch

from representation_learning.configs import AudioConfig
from representation_learning.models.eat.audio_model import Model as EATModel


def test_eat_wrapper_forward() -> None:
    """Minimal smoke test for the EAT audio wrapper.

    Ensures that the model can run a forward pass on CPU and returns logits of
    the expected shape.
    """
    batch_size = 2
    sample_rate = 16000
    duration_seconds = 1
    wav_len = sample_rate * duration_seconds

    dummy_wave = torch.randn(batch_size, wav_len)

    # Use a mel height compatible with the EAT patch embedding
    model = EATModel(
        num_classes=10,
        device="cpu",
        audio_config=AudioConfig(
            sample_rate=sample_rate, n_mels=128, n_fft=2048, hop_length=512
        ),
        target_length=128,
    )

    logits = model(dummy_wave, padding_mask=None)
    assert logits.shape == (batch_size, 10)
