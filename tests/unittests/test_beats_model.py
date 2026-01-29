"""Unit tests for BEATs model.

Tests the BEATs model using official models from Hugging Face Hub.
"""

import torch

from avex import load_model


def test_beats_model_forward() -> None:
    """Smoke-test for the BEATs model using official HuggingFace checkpoint.

    Uses the esp_aves2_sl_beats_all model which is hosted on Hugging Face Hub.
    """
    batch_size = 2
    num_samples = 160_000  # 10-second audio at 16 kHz (matches model's target_length_seconds)

    # Random noise as dummy audio
    audio = torch.randn(batch_size, num_samples)

    # Load official BEATs model from HuggingFace Hub in embedding extraction mode
    model = load_model("esp_aves2_sl_beats_all", device="cpu", return_features_only=True)

    # Forward pass
    with torch.no_grad():
        # BEATs expects (batch, samples) input
        features = model(audio)

    # Assertions - BEATs returns (batch, time_steps, 768) embeddings
    assert features.dim() == 3
    assert features.shape[0] == batch_size
    assert features.shape[2] == 768  # BEATs embedding dimension
    assert not torch.isnan(features).any()
