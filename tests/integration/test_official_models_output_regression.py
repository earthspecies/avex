"""Regression tests for official model checkpoint outputs on labeled audio.

This test complements checksum verification by asserting that official checkpoints
also produce stable numerical outputs on a deterministic labeled mini-batch.
"""

from __future__ import annotations

import hashlib
import sys
from typing import Final

import numpy as np
import pytest
import torch

from avex import load_model
from avex.models.utils.registry import get_checkpoint_path, list_models

# Expected pooled-output fingerprints from deterministic labeled mini-batch.
# Fingerprint is SHA-256 of np.round(output, 4).tobytes().
#
# Different Python bands can share one table when the locked numerical stack
# (torch, numpy, tensorflow, etc.) yields identical rounded outputs. When a new
# Python or dependency bump changes numerics, add a new profile key below and
# map it in _fingerprint_profile() — no per-minor-version fixture files.
_OFFICIAL_MODEL_OUTPUT_FINGERPRINTS_BY_PROFILE: Final[dict[str, dict[str, str]]] = {
    "py310_312": {
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
    },
    # Python 3.13+ with tensorflow>=2.21, torch 2.11.x stack (see uv.lock on branch).
    "py313_plus": {
        "esp_aves2_eat_all": "c4b84d7f28b6d4fee28702c1d051aecb1109b272028926752cac0ee19df979b5",
        "esp_aves2_eat_bio": "c4b84d7f28b6d4fee28702c1d051aecb1109b272028926752cac0ee19df979b5",
        "esp_aves2_effnetb0_all": "801ebab010118dd0f07f0a07d0f18f5aa64f2f1270fe4bc2123342c081fa1b53",
        "esp_aves2_effnetb0_audioset": "1fbe57dd3b795aea08c66ed5c45731cce5a08835b9c33676057d6e5d361c52ae",
        "esp_aves2_effnetb0_bio": "3123856a920e27271a26fe29437119a90e2ebd5436d4bb9a4629d08828fef8ef",
        "esp_aves2_naturelm_audio_v1_beats": "f747f0f5b8590253b600eb04ae369c1440e4ed1be5ca65515d296f4f88d447b8",
        "esp_aves2_sl_beats_all": "930e2b1ed2168db90ba4279f3ca3563bf77f81559906d6a43c9951503da73c21",
        "esp_aves2_sl_beats_bio": "f5fc0b8815267b7ae8fbb3adbadcb3caa85f970f7f1abe1ec746e069228b3b97",
        "esp_aves2_sl_eat_all_ssl_all": "cc21667c9aca79fc2fb946685b28c7ef5dfee587181297d52fc5110832e05616",
        "esp_aves2_sl_eat_bio_ssl_all": "61f01e24bfa200f109063bd68af78ad9fd38a0d5422a12919e65efd263a12ecf",
    },
}


def _fingerprint_profile() -> str:
    """Return which fingerprint table applies to this interpreter.

    Returns:
        Key into ``_OFFICIAL_MODEL_OUTPUT_FINGERPRINTS_BY_PROFILE``. Extend this
        when a new Python line or stack bump diverges from existing bands.
    """
    if sys.version_info < (3, 13):
        return "py310_312"
    return "py313_plus"


def _expected_official_output_fingerprints() -> dict[str, str]:
    """Fingerprints for the current runtime profile.

    Returns:
        Model name to expected SHA-256 fingerprint for
        ``_fingerprint_profile()``.
    """
    return _OFFICIAL_MODEL_OUTPUT_FINGERPRINTS_BY_PROFILE[_fingerprint_profile()]


def _validate_fingerprint_profile_tables() -> None:
    """Ensure every profile defines the same set of model names."""
    profiles = list(_OFFICIAL_MODEL_OUTPUT_FINGERPRINTS_BY_PROFILE.items())
    reference_name, reference_table = profiles[0]
    reference_keys = set(reference_table.keys())
    for name, table in profiles[1:]:
        keys = set(table.keys())
        assert keys == reference_keys, (
            f"Fingerprint keys for profile {name!r} differ from {reference_name!r}: "
            f"only-in-first={sorted(reference_keys - keys)} "
            f"only-in-second={sorted(keys - reference_keys)}"
        )


_validate_fingerprint_profile_tables()

# Same keys in every profile (enforced above); any profile works for parametrization.
_OFFICIAL_MODEL_NAMES_FOR_OUTPUT_REGRESSION: Final[tuple[str, ...]] = tuple(
    sorted(_OFFICIAL_MODEL_OUTPUT_FINGERPRINTS_BY_PROFILE["py310_312"].keys())
)

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
        expected = set(_expected_official_output_fingerprints().keys())
        assert expected == official, (
            "Fingerprint table mismatch. Update _OFFICIAL_MODEL_OUTPUT_FINGERPRINTS_BY_PROFILE to "
            f"exactly match official HF models.\nExpected-only: {sorted(expected - official)}\n"
            f"Official-only: {sorted(official - expected)}"
        )

    @pytest.mark.parametrize("model_name", _OFFICIAL_MODEL_NAMES_FOR_OUTPUT_REGRESSION)
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
        expected_digest = _expected_official_output_fingerprints()[model_name]

        assert digest == expected_digest, (
            f"Output fingerprint mismatch for {model_name}. "
            f"Expected {expected_digest}, got {digest}. "
            "If checkpoint changed intentionally, regenerate and update reference."
        )
