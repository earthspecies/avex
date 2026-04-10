"""Regression tests for official model checkpoint outputs on labeled audio.

This test complements checksum verification by asserting that official checkpoints
also produce stable numerical outputs on a deterministic labeled mini-batch.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from avex import load_model
from avex.models.utils.registry import get_checkpoint_path, list_models

# Expected pooled-output fingerprints from deterministic labeled mini-batch.
# Fingerprint is SHA-256 of np.round(output, 4).tobytes().
OFFICIAL_MODEL_OUTPUT_FINGERPRINTS: dict[str, str] = {
    "esp_aves2_eat_all": "d5d462c560352c1c3c9f498a0951f56ec9924e50f8fe1f0f0a4d285e316c17c8",
    "esp_aves2_eat_bio": "d5d462c560352c1c3c9f498a0951f56ec9924e50f8fe1f0f0a4d285e316c17c8",
    "esp_aves2_effnetb0_all": "7f1e8cc046287f79a3a2b7413042ff121a3f32c115cf3a487d2b5348e09a4931",
    "esp_aves2_effnetb0_audioset": "8ba36f99b5e8245d7b61fc472339f5760fabca19d63a51e835309c11a379eab6",
    "esp_aves2_effnetb0_bio": "c91dde6bee57788951a0fb9044703d301cb295e83fdc5e064874b63c99c70493",
    "esp_aves2_naturelm_audio_v1_beats": "c1689532213d32cc16b0f7eb1774239c4d4bbd91a0500b551d4468acf52cb9d1",
    "esp_aves2_sl_beats_all": "b6231fdcb855734ebfddf26e793a46d8e4b3bf61ee950273fdd85affcf85eefe",
    "esp_aves2_sl_beats_bio": "1ad22272d36f3e74d64c5fb98ec31810c9281c1c32e9a2178f10c08004c8bcd6",
    "esp_aves2_sl_eat_all_ssl_all": "0832f0c78523167e0a5439b9a4e96caf115131118549ff9161a01bd6d03a5b2e",
    "esp_aves2_sl_eat_bio_ssl_all": "a9302a12a55bb6c1379b2dc42a22c15150eab12d039f7ad8c8d793a5dc31af70",
}

_HF_PREFIX = "hf://"


def _official_hf_model_names() -> list[str]:
    """Return official ESP model names with HF-backed checkpoints.

    Returns:
        Sorted list of registry model names whose checkpoint paths start with
        the Hugging Face URI prefix.
    """
    names: list[str] = []
    for model_name in list_models().keys():
        if not model_name.startswith("esp_"):
            continue
        checkpoint_path = get_checkpoint_path(model_name)
        if checkpoint_path is not None and checkpoint_path.startswith(_HF_PREFIX):
            names.append(model_name)
    return sorted(names)


def _build_labeled_audio_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic labeled mini-batch with three synthetic classes.

    Returns:
        Tuple of `(audio, labels)` where audio has shape `(6, 16000)` and labels
        has shape `(6,)`.
    """
    sample_rate = 16_000
    # Discrete-time grid: 16000 samples at 16kHz for 1 second (endpoint excluded).
    t = torch.arange(sample_rate, dtype=torch.float32) / float(sample_rate)
    freqs = (220.0, 440.0, 880.0)

    clips: list[torch.Tensor] = []
    labels: list[int] = []
    for class_index, freq in enumerate(freqs):
        base = torch.sin(2.0 * torch.pi * freq * t)
        for amplitude in (0.8, 0.9):
            clips.append((amplitude * base).to(torch.float32))
            labels.append(class_index)

    expected_labels = torch.tensor(labels, dtype=torch.long)
    return torch.stack(clips, dim=0), expected_labels


def _pooled_model_output(model_name: str, audio: torch.Tensor) -> torch.Tensor:
    """Load model and produce pooled clip-level outputs.

    Args:
        model_name: Official model registry key.
        audio: Input batch shaped `(B, T)`.

    Returns:
        Tensor shaped `(B, D)` after temporal/spatial pooling when needed.

    Raises:
        ValueError: If model output tensor rank is not 2, 3, or 4.
    """
    model = load_model(model_name, device="cpu", return_features_only=True)
    model.eval()

    with torch.no_grad():
        output = model(audio)

    if output.dim() == 2:
        return output
    if output.dim() == 3:
        return output.mean(dim=1)
    if output.dim() == 4:
        return output.mean(dim=(2, 3))
    raise ValueError(f"Unsupported output rank for {model_name}: shape={tuple(output.shape)}")


@pytest.mark.slow
class TestOfficialModelsOutputRegression:
    """Regression tests for official model numerical output stability."""

    @pytest.fixture(autouse=True)
    def _ensure_registry_initialized(self) -> None:
        """Ensure model registry is populated before each test."""
        from avex.models.utils import registry

        registry.initialize_registry()

    def test_reference_table_covers_all_official_hf_models(self) -> None:
        """Ensure every official HF model has an expected output fingerprint."""
        official = set(_official_hf_model_names())
        expected = set(OFFICIAL_MODEL_OUTPUT_FINGERPRINTS.keys())
        assert expected == official, (
            "Fingerprint table mismatch. Update OFFICIAL_MODEL_OUTPUT_FINGERPRINTS to "
            f"exactly match official HF models.\nExpected-only: {sorted(expected - official)}\n"
            f"Official-only: {sorted(official - expected)}"
        )

    @pytest.mark.parametrize("model_name", sorted(OFFICIAL_MODEL_OUTPUT_FINGERPRINTS.keys()))
    def test_official_model_output_matches_expected_fingerprint(self, model_name: str) -> None:
        """Assert model output fingerprint matches expected reference value.

        Raises
        ------
        ValueError
            If the model returns an output tensor with unsupported rank.
        """
        audio, labels = _build_labeled_audio_batch()
        assert labels.shape[0] == audio.shape[0], "Labeled batch must align audio and labels."
        expected_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        assert torch.equal(labels, expected_labels), "Labeled batch must have stable, expected labels."

        try:
            model = load_model(model_name, device="cpu", return_features_only=True)
        except (OSError, ConnectionError, TimeoutError) as exc:  # pragma: no cover - network/model availability
            pytest.skip(f"Unable to load model {model_name!r}: {exc}")

        model.eval()
        with torch.no_grad():
            output = model(audio)

        if output.dim() == 2:
            pooled = output
        elif output.dim() == 3:
            pooled = output.mean(dim=1)
        elif output.dim() == 4:
            pooled = output.mean(dim=(2, 3))
        else:
            raise ValueError(f"Unsupported output rank for {model_name}: shape={tuple(output.shape)}")

        pooled_np = pooled.detach().cpu().to(torch.float32).numpy()
        rounded = np.round(pooled_np, 4)
        digest = hashlib.sha256(rounded.tobytes()).hexdigest()
        expected_digest = OFFICIAL_MODEL_OUTPUT_FINGERPRINTS[model_name]

        assert digest == expected_digest, (
            f"Output fingerprint mismatch for {model_name}. "
            f"Expected {expected_digest}, got {digest}. "
            "If checkpoint changed intentionally, regenerate and update reference."
        )
