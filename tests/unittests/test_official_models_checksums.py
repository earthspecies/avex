"""Tests that verify checksums of official models hosted on the Hugging Face Hub.

Expected SHA-256 hashes are hardcoded below. When official models are (re-)uploaded,
update OFFICIAL_MODEL_CHECKSUMS from the .safetensors.sha256 files produced in the
upload folders (or from the same files in each HF repo).
"""

from __future__ import annotations

import hashlib
from importlib import resources

import pytest

from representation_learning.io import filesystem_from_path
from representation_learning.models.utils.registry import get_checkpoint_path

# Expected SHA-256 (hex) of each official model's safetensors file.
# Keys = official model name (YAML stem); values = 64-char hex lowercase.
# Update from the .safetensors.sha256 files when models are (re-)uploaded.
OFFICIAL_MODEL_CHECKSUMS: dict[str, str] = {
    "esp_aves2_eat_all": "",
    "esp_aves2_eat_bio": "",
    "esp_aves2_effnetb0_all": "60a54b874b0ae9a3a3fec9143a6b73aaf59f2e77c20d19fe1fabc797da3faeb1",
    "esp_aves2_effnetb0_audioset": "",
    "esp_aves2_effnetb0_bio": "",
    "esp_aves2_naturelm_audio_v1_beats": "",
    "esp_aves2_sl_beats_all": "",
    "esp_aves2_sl_beats_bio": "",
    "esp_aves2_sl_eat_all_ssl_all": "",
    "esp_aves2_sl_eat_bio_ssl_all": "",
}

# Prefix for Hugging Face Hub checkpoint paths (only these are checked).
_HF_PREFIX = "hf://"


def _official_hf_model_names() -> list[str]:
    """Return official model names that have an hf:// checkpoint path.

    Returns
    -------
    list[str]
        Sorted list of official model names whose checkpoint_path is on the Hub.
    """
    pkg = "representation_learning.api.configs.official_models"
    root = resources.files(pkg)
    names: list[str] = []
    for entry in root.iterdir():
        if not entry.name.endswith(".yml") or not entry.is_file():
            continue
        name = entry.stem
        try:
            checkpoint_path = get_checkpoint_path(name)
            if checkpoint_path and checkpoint_path.startswith(_HF_PREFIX):
                names.append(name)
        except KeyError:
            continue
    return sorted(names)


def _path_for_hf_fs(path: str) -> str:
    """Return the path string to pass to HfFileSystem.open().

    HfFileSystem expects repo_id/path_in_repo (no hf://).

    Parameters
    ----------
    path : str
        Full path, possibly with hf:// prefix.

    Returns
    -------
    str
        Path with hf:// stripped, or unchanged if not an hf path.
    """
    if path.startswith(_HF_PREFIX):
        return path[len(_HF_PREFIX) :]
    return path


class TestOfficialModelsChecksumsHardcoded:
    """Test that hardcoded checksums are valid hex (no network)."""

    def test_all_official_hf_models_have_checksum_entry(self) -> None:
        """Every official HF model must have an entry in OFFICIAL_MODEL_CHECKSUMS."""
        for name in _official_hf_model_names():
            assert name in OFFICIAL_MODEL_CHECKSUMS, (
                f"Add OFFICIAL_MODEL_CHECKSUMS[{name!r}] from the model's "
                ".safetensors.sha256 file (upload folder or HF repo)."
            )

    def test_checksum_entries_are_valid_hex_when_set(self) -> None:
        """When set, each checksum must be 64 lowercase hex characters."""
        for name, hex_val in OFFICIAL_MODEL_CHECKSUMS.items():
            if not hex_val:
                continue
            assert len(hex_val) == 64, f"OFFICIAL_MODEL_CHECKSUMS[{name!r}] must be 64 chars"
            assert all(c in "0123456789abcdef" for c in hex_val), (
                f"OFFICIAL_MODEL_CHECKSUMS[{name!r}] must be lowercase hex"
            )


@pytest.mark.slow
class TestOfficialModelsSafetensorsMatchHardcodedChecksum:
    """Test that each safetensors file on the Hub matches the hardcoded SHA-256.

    Skips models with no checksum set. Run with: pytest -m slow ...
    """

    @pytest.mark.parametrize("model_name", _official_hf_model_names())
    def test_safetensors_matches_hardcoded_checksum(self, model_name: str) -> None:
        """Download safetensors and verify its SHA-256 matches OFFICIAL_MODEL_CHECKSUMS."""
        expected_hex = OFFICIAL_MODEL_CHECKSUMS.get(model_name, "").strip()
        if not expected_hex:
            pytest.skip(
                f"No checksum set for {model_name}; add from .safetensors.sha256 and update OFFICIAL_MODEL_CHECKSUMS."
            )

        checkpoint_path = get_checkpoint_path(model_name)
        assert checkpoint_path is not None
        assert checkpoint_path.startswith(_HF_PREFIX)

        fs = filesystem_from_path(checkpoint_path)
        path_for_fs = _path_for_hf_fs(checkpoint_path)
        h = hashlib.sha256()
        with fs.open(path_for_fs, mode="rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual_hex = h.hexdigest().lower()

        assert actual_hex == expected_hex, (
            f"Checksum mismatch for {model_name}: "
            f"expected {expected_hex}, got {actual_hex}. "
            "Re-upload may have changed the file; update OFFICIAL_MODEL_CHECKSUMS."
        )
