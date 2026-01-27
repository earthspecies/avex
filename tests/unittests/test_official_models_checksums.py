"""Tests that verify checksums of official models hosted on the Hugging Face Hub.

Expected SHA-256 hashes are hardcoded below. When official models are (re-)uploaded,
update OFFICIAL_MODEL_CHECKSUMS from the .safetensors.sha256 files produced in the
upload folders (or from the same files in each HF repo).
"""

from __future__ import annotations

import hashlib
import re
from importlib import resources

import pytest

from representation_learning.io import filesystem_from_path
from representation_learning.models.utils.registry import get_checkpoint_path

# Expected SHA-256 (hex) of each official model's safetensors file.
# Keys = official model name (YAML stem); values are 64-digit lowercase hex.
# Update from the .safetensors.sha256 files when official models are (re-)uploaded.
OFFICIAL_MODEL_CHECKSUMS: dict[str, str] = {
    "esp_aves2_eat_all": "8dcd5f9059785f7a2b1ff19b81ddcf75bab80a359a04f38a9da728fd33b25d39",
    "esp_aves2_eat_bio": "8968be9d42668d5f62da47e3ea725258ec53195544434d5f62adc856f04e3639",
    "esp_aves2_effnetb0_all": "60a54b874b0ae9a3a3fec9143a6b73aaf59f2e77c20d19fe1fabc797da3faeb1",
    "esp_aves2_effnetb0_audioset": "a17360719494663ac12372adc1de0beee296036349ed603f939ef455535c77a3",
    "esp_aves2_effnetb0_bio": "1b74aab200e2035582da47e3ea725258ec53195544434d5f62adc856f04e3639",
    "esp_aves2_naturelm_audio_v1_beats": "730d2c3b6c67d3e0a26ef9d5c3b7d0fbb20a7dca104456922a6b16e6b7a696b8",
    "esp_aves2_sl_beats_all": "a928a74190e2d314e3f146805a7c76772a91cce75b4101a88fde845a662a2cc7",
    "esp_aves2_sl_beats_bio": "1881788eb6d059d7b005e1c68235906fcb12bf3a6cde824cec7cbdc34dcb9fc3",
    "esp_aves2_sl_eat_all_ssl_all": "af10ff12eb15b0e134334d787b4ccb97bd3e4fe11147140c68ba646d64130cc",
    "esp_aves2_sl_eat_bio_ssl_all": "d787a181898e4ca68e0d0fa78dc2de83b27c2bd1648bce476534fc8c5ac2c7d",
}

# Prefix for Hugging Face Hub checkpoint paths (only these are checked).
_HF_PREFIX = "hf://"

# .safetensors.sha256 lines look like: "<64-hex>  filename"
_SHA256_LINE_RE = re.compile(r"^([0-9a-fA-F]{64})\\s+\\S+\\s*$")


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

    @pytest.mark.parametrize("model_name", _official_hf_model_names())
    def test_hardcoded_checksum_matches_hf_sha256(self, model_name: str) -> None:
        """Ensure hardcoded checksum matches the .safetensors.sha256 file on HF.

        This is a fast sanity check: it only reads the small `.safetensors.sha256`
        sidecar file (not the full weights) and compares its hex digest against
        the hardcoded value in OFFICIAL_MODEL_CHECKSUMS.
        """
        expected_hex = OFFICIAL_MODEL_CHECKSUMS.get(model_name, "").strip()
        assert expected_hex, (
            f"OFFICIAL_MODEL_CHECKSUMS[{model_name!r}] is empty; populate it from the model's .safetensors.sha256 file."
        )

        checkpoint_path = get_checkpoint_path(model_name)
        assert checkpoint_path is not None
        assert checkpoint_path.startswith(_HF_PREFIX)

        sha256_path = checkpoint_path + ".sha256"
        fs = filesystem_from_path(sha256_path)
        path_for_fs = _path_for_hf_fs(sha256_path)
        try:
            with fs.open(path_for_fs, mode="r", encoding="utf-8") as f:
                line = f.readline().strip()
        except Exception as exc:  # pragma: no cover - network / HF issues
            pytest.skip(f"Could not read {sha256_path!r} from HF: {exc}")

        match = _SHA256_LINE_RE.match(line)
        assert match is not None, f"Invalid .sha256 format for {model_name}: {line!r}"
        hf_hex = match.group(1).lower()
        assert hf_hex == expected_hex, (
            f"Hardcoded checksum for {model_name} does not match HF .sha256: expected {expected_hex}, got {hf_hex}"
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
