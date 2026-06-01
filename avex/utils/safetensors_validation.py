"""Validation helpers for published safetensors checkpoint artifacts."""

from __future__ import annotations

from pathlib import Path

from safetensors import safe_open

MIN_PUBLISHED_SAFETENSORS_BYTES = 1024
MIN_PUBLISHED_SAFETENSORS_TENSORS = 1


class SafetensorsWeightsError(ValueError):
    """Raised when a safetensors artifact does not contain model weights."""


def assert_safetensors_has_weights(
    path: Path | str,
    *,
    min_bytes: int = MIN_PUBLISHED_SAFETENSORS_BYTES,
    min_tensors: int = MIN_PUBLISHED_SAFETENSORS_TENSORS,
) -> int:
    """Ensure a local safetensors file contains a non-empty tensor map.

    Parameters
    ----------
    path
        Local path to the safetensors artifact.
    min_bytes
        Minimum on-disk file size required for published checkpoints.
    min_tensors
        Minimum number of tensors required in the safetensors payload.

    Returns
    -------
    int
        On-disk file size in bytes.

    Raises
    ------
    SafetensorsWeightsError
        If the artifact is too small or contains no tensors.
    FileNotFoundError
        If the artifact path does not exist.
    """
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Safetensors file not found: {artifact_path}")

    size_bytes = artifact_path.stat().st_size
    if size_bytes < min_bytes:
        raise SafetensorsWeightsError(
            f"Safetensors artifact is too small ({size_bytes} bytes < {min_bytes} bytes): {artifact_path}"
        )

    with safe_open(str(artifact_path), framework="pt", device="cpu") as handle:
        tensor_keys = tuple(handle.keys())

    if len(tensor_keys) < min_tensors:
        raise SafetensorsWeightsError(
            f"Safetensors artifact contains no model weights "
            f"({len(tensor_keys)} tensors < {min_tensors}): {artifact_path}"
        )

    return size_bytes
