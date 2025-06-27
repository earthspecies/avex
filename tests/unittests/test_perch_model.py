from __future__ import annotations

import pytest
import torch

# Import the module under test *before* monkey-patching so pytest can access it.
import representation_learning.models.perch as perch_module


def test_perch_model_forward_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke-test PerchModel forward pass on CPU without TensorFlow.

    The real Perch backbone is implemented in TensorFlow. To keep the unit test
    lightweight and runnable in CI jobs that do not have TensorFlow installed,
    we monkey-patch both the TF-loading routine and the private `_tf_forward`
    method to return deterministic dummy embeddings.  This allows us to verify
    that the PyTorch wrapper (device transfer, classifier head, etc.) works and
    that output shapes are correct.
    """

    # ------------------------------------------------------------------
    #  Monkey-patch TensorFlow-specific parts
    # ------------------------------------------------------------------

    # 1) Replace the lazy TF-Hub loader with a no-op.
    monkeypatch.setattr(perch_module, "_load_tf_model", lambda: None, raising=True)

    # 2) Stub `_tf_forward` so it returns zeros with the expected embedding dim.
    EMB_DIM = 1280

    def _dummy_tf_forward(
        self: perch_module.PerchModel, audio: torch.Tensor
    ) -> torch.Tensor:  # noqa: D401,E501
        batch = audio.shape[0]
        return torch.zeros(batch, EMB_DIM)

    monkeypatch.setattr(
        perch_module.PerchModel, "_tf_forward", _dummy_tf_forward, raising=True
    )

    # ------------------------------------------------------------------
    #  Create dummy waveform and run the model
    # ------------------------------------------------------------------
    batch_size = 2
    sample_rate = 32_000
    duration_seconds = 5
    wav_len = sample_rate * duration_seconds

    dummy_audio = torch.randn(batch_size, wav_len)

    model = perch_module.PerchModel(
        num_classes=10,
        device="cpu",
    )

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_audio, padding_mask=None)

    # Assertions
    assert logits.shape == (batch_size, 10)
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(
    pytest.importorskip("tensorflow", reason="TensorFlow required for real Perch test")
    is None,
    reason="TensorFlow not installed",
)
def test_perch_model_forward_real() -> None:
    """Integration test that exercises the *real* TensorFlow backbone.

    The test is skipped automatically if TensorFlow or TF-Hub cannot be
    imported (e.g. lightweight CI runners).  When dependencies are present it
    downloads the Perch TF-Hub module, runs a forward pass, and checks output
    integrity.
    """

    # Import inside the test after the skip condition so that pytest.importorskip
    # can do its job without side effects in environments lacking TF.
    import representation_learning.models.perch as perch_module

    batch_size = 1  # keep minimal for quick CI
    wav_len = 32_000 * 5  # 5-second clip at 32 kHz

    dummy_audio = torch.randn(batch_size, wav_len)

    try:
        model = perch_module.PerchModel(
            num_classes=0,  # embeddings only – no classifier head
            device="cpu",
        )
    except Exception as e:
        # If TF-Hub download fails (e.g. no network) skip instead of failing CI.
        print(f"Maybe skipping real Perch test – maybe could not load TF model: {e}")
        # pytest.skip(f"Skipping real Perch test – could not load TF model: {e}")

    with torch.no_grad():
        emb = model(dummy_audio, padding_mask=None)

    # Embedding shape should be (B, 1280)
    assert emb.shape == (batch_size, 1280)
    assert torch.isfinite(emb).all()
