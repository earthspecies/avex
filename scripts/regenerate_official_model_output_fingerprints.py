"""Regenerate expected output fingerprints for official ESP HF models.

This utility builds the same deterministic labeled mini-batch used by
`tests/integration/test_official_models_output_regression.py`, runs all official
HF-backed models in feature mode, and prints a Python snippet for the selected
**profile** inside ``_OFFICIAL_MODEL_OUTPUT_FINGERPRINTS_BY_PROFILE`` (bands
like ``py310_312`` vs ``py313_plus``, not one file per Python minor).

Usage:
    # From a 3.12 venv (updates the py310_312 band):
    uv run python scripts/regenerate_official_model_output_fingerprints.py --profile py310_312

    # From a 3.13+ venv (updates the py313_plus band):
    uv run python scripts/regenerate_official_model_output_fingerprints.py --profile py313_plus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import torch

from avex import load_model
from avex.models.utils.registry import get_checkpoint_path, list_models

_HF_PREFIX = "hf://"

_VALID_PROFILES: tuple[str, ...] = ("py310_312", "py313_plus")


def _profile_from_interpreter() -> str:
    """Match ``tests/integration/test_official_models_output_regression.py``.

    Returns:
        ``py310_312`` when ``sys.version_info < (3, 13)``, otherwise
        ``py313_plus``.
    """
    if sys.version_info < (3, 13):
        return "py310_312"
    return "py313_plus"


def _official_hf_model_names() -> list[str]:
    """Return official ESP model names with HF-backed checkpoints.

    Returns:
        Sorted official model names.
    """
    names: list[str] = []
    for model_name in list_models().keys():
        if not model_name.startswith("esp_"):
            continue
        checkpoint_path = get_checkpoint_path(model_name)
        if checkpoint_path is not None and checkpoint_path.startswith(_HF_PREFIX):
            names.append(model_name)
    return sorted(names)


def _build_labeled_audio_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic labeled mini-batch with three synthetic classes.

    Args:
        seed: Torch seed used for deterministic setup.

    Returns:
        Tuple of `(audio, labels)` with shapes `(6, 16000)` and `(6,)`.
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


def _pool_output(output: torch.Tensor, model_name: str) -> torch.Tensor:
    """Pool model outputs to clip-level shape `(B, D)`.

    Args:
        output: Raw model output tensor.
        model_name: Name used for error messages.

    Returns:
        Pooled output tensor.

    Raises:
        ValueError: If ``output`` rank is not 2, 3, or 4.
    """
    if output.dim() == 2:
        return output
    if output.dim() == 3:
        return output.mean(dim=1)
    if output.dim() == 4:
        return output.mean(dim=(2, 3))
    raise ValueError(f"Unsupported output rank for {model_name}: shape={tuple(output.shape)}")


def _fingerprint_output(output: torch.Tensor, decimals: int) -> str:
    """Compute SHA-256 fingerprint from rounded float output.

    Args:
        output: Pooled output tensor.
        decimals: Number of decimal places for rounding.

    Returns:
        Hex SHA-256 digest string.
    """
    array = output.detach().cpu().to(torch.float32).numpy()
    rounded = np.round(array, decimals=decimals)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def _compute_fingerprints(
    model_names: list[str],
    audio: torch.Tensor,
    decimals: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Compute per-model fingerprints and capture load/run errors.

    Args:
        model_names: Official model names to evaluate.
        audio: Input audio batch.
        decimals: Number of decimal places for rounding.

    Returns:
        Tuple of `(fingerprints, errors)`.
    """
    fingerprints: dict[str, str] = {}
    errors: dict[str, str] = {}

    for model_name in model_names:
        try:
            model = load_model(model_name, device="cpu", return_features_only=True)
            model.eval()
            with torch.no_grad():
                output = model(audio)
            pooled = _pool_output(output, model_name=model_name)
            fingerprints[model_name] = _fingerprint_output(pooled, decimals=decimals)
        except Exception as exc:  # pragma: no cover - depends on environment/network
            errors[model_name] = str(exc)
    return fingerprints, errors


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Regenerate official model output fingerprints.")
    parser.add_argument(
        "--profile",
        choices=_VALID_PROFILES,
        default=None,
        help=(
            "Fingerprint band to print (must match a key in "
            "_OFFICIAL_MODEL_OUTPUT_FINGERPRINTS_BY_PROFILE). "
            "Default: infer from running Python (3.10–3.12 → py310_312, else py313_plus)."
        ),
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Rounding decimals before hashing (default: 4).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of Python dict literal.",
    )
    return parser.parse_args()


def main() -> int:
    """Run fingerprint generation and print output.

    Returns:
        ``0`` if every model produced a fingerprint, ``1`` if any model failed
        and was recorded in the errors map.

    Raises:
        ValueError: If the labeled audio batch and labels have mismatched lengths.
    """
    args = parse_args()
    profile = args.profile if args.profile is not None else _profile_from_interpreter()
    model_names = _official_hf_model_names()
    audio, labels = _build_labeled_audio_batch(seed=7)
    if labels.shape[0] != audio.shape[0]:
        raise ValueError("Labeled batch mismatch between audio and labels.")

    fingerprints, errors = _compute_fingerprints(model_names, audio=audio, decimals=args.decimals)

    if args.json:
        print(json.dumps({"profile": profile, "fingerprints": fingerprints}, indent=2, sort_keys=True))
    else:
        print(
            f"# Paste/replace the inner dict for profile {profile!r} in "
            "tests/integration/test_official_models_output_regression.py"
        )
        print(f'    "{profile}": {{')
        for name in sorted(fingerprints.keys()):
            print(f'        "{name}": "{fingerprints[name]}",')
        print("    },")

    if errors:
        print("\nErrors (models skipped):")
        for name in sorted(errors.keys()):
            print(f"- {name}: {errors[name]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
