import torch

from avex.models.beats_model import Model as BeatsModel

# ---------------------------------------------------------------------------
#  Now that stubs are in place, we can safely import the model under test.
# ---------------------------------------------------------------------------


def test_beats_model_forward() -> None:
    """Simple smoke-test for the BEATs wrapper on CPU."""

    batch_size = 2
    num_samples = 16_000  # 1-second audio at 16 kHz
    num_classes = 10

    # Random noise as dummy audio
    audio = torch.randn(batch_size, num_samples)

    # All-ones mask (no padding). Shape must match (batch, time)
    padding_mask = torch.zeros(batch_size, num_samples, dtype=torch.bool)

    model = BeatsModel(
        num_classes=num_classes,
        pretrained=False,
        device="cpu",
    )

    # Forward pass
    with torch.no_grad():
        logits = model(audio, padding_mask)

    # Assertions
    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()
