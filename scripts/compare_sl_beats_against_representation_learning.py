"""Compare SL-BEATs models between `avex` and `representation_learning`.

This script is intended as a lightweight regression/debug tool:
- Load the same model key from both libraries.
- Compare state_dict key overlap and strict tensor equality on shared keys.
- Run a deterministic forward pass on synthetic audio and compare outputs.

Example
-------
uv run python scripts/compare_sl_beats_against_representation_learning.py
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class _Loaded:
    name: str
    model: Any
    state: dict[str, torch.Tensor]


def _find_representation_learning_load_model() -> Callable[..., Any]:
    """Best-effort discovery of representation_learning's `load_model` API.

    Returns
    -------
    Callable[..., Any]
        Callable implementing the `load_model` API.

    Raises
    ------
    RuntimeError
        If no supported `load_model` symbol can be found.
    """
    candidates: list[tuple[str, str]] = [
        ("representation_learning.models.utils.load", "load_model"),
        ("representation_learning.models.utils.registry", "load_model"),
        ("representation_learning.models.load", "load_model"),
        ("representation_learning.api", "load_model"),
    ]
    for module_name, fn_name in candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(mod, fn_name):
            return getattr(mod, fn_name)
    raise RuntimeError(
        "Could not locate `load_model` in representation_learning. Add the correct import path to candidates."
    )


def _load_avex(model_key: str) -> _Loaded:
    from avex.models.utils.load import load_model

    model = load_model(model_key, device="cpu", return_features_only=True)
    model.eval()
    return _Loaded(name=f"avex:{model_key}", model=model, state=model.state_dict())


def _load_representation_learning(model_key: str) -> _Loaded:
    load_model = _find_representation_learning_load_model()
    model = load_model(model_key, device="cpu", return_features_only=True)
    model.eval()
    return _Loaded(
        name=f"representation_learning:{model_key}",
        model=model,
        state=model.state_dict(),
    )


def _strict_state_dict_compare(a: _Loaded, b: _Loaded) -> None:
    a_keys = set(a.state.keys())
    b_keys = set(b.state.keys())
    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)
    common = sorted(a_keys & b_keys)

    print(f"{a.name} tensors: {len(a_keys)}")
    print(f"{b.name} tensors: {len(b_keys)}")
    print(f"common_tensors: {len(common)}")

    if only_a:
        print(f"\nOnly in {a.name} ({len(only_a)}):")
        for k in only_a:
            t = a.state[k]
            print(f"  - {k}  shape={tuple(t.shape)} dtype={t.dtype}")

    if only_b:
        print(f"\nOnly in {b.name} ({len(only_b)}):")
        for k in only_b:
            t = b.state[k]
            print(f"  - {k}  shape={tuple(t.shape)} dtype={t.dtype}")

    shape_mismatch: list[str] = []
    dtype_mismatch: list[str] = []
    value_mismatch: list[str] = []

    for k in common:
        ta = a.state[k]
        tb = b.state[k]
        if ta.shape != tb.shape:
            shape_mismatch.append(k)
            continue
        if ta.dtype != tb.dtype:
            dtype_mismatch.append(k)
            continue
        if not torch.equal(ta, tb):
            value_mismatch.append(k)

    print("\nStrict equality on common keys")
    print("shape_mismatch:", len(shape_mismatch))
    print("dtype_mismatch:", len(dtype_mismatch))
    print("value_mismatch:", len(value_mismatch))
    if value_mismatch:
        print("mismatching_keys:")
        for k in value_mismatch:
            print(f"  - {k}")


def _deterministic_forward_compare(a: _Loaded, b: _Loaded) -> None:
    torch.manual_seed(123)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    # 1 second @ 16kHz, batch 2
    audio = torch.randn(2, 16_000, dtype=torch.float32)

    with torch.no_grad():
        out_a = a.model(audio)
        out_b = b.model(audio)

    if out_a.shape != out_b.shape:
        raise AssertionError(f"Output shape mismatch: {a.name} {tuple(out_a.shape)} vs {b.name} {tuple(out_b.shape)}")

    diff = (out_a.float() - out_b.float()).abs()
    print("\nDeterministic forward comparison (same seed, same input)")
    print("output_shape:", tuple(out_a.shape))
    print("exact_match:", torch.equal(out_a, out_b))
    print("max_abs_diff:", float(diff.max().item()))
    print("mean_abs_diff:", float(diff.mean().item()))


def main() -> None:
    model_keys = [
        "esp_aves2_sl_beats_all",
        "esp_aves2_sl_beats_bio",
    ]

    for model_key in model_keys:
        print("\n==============================")
        print("MODEL:", model_key)
        print("==============================")

        av = _load_avex(model_key)
        rl = _load_representation_learning(model_key)

        _strict_state_dict_compare(av, rl)
        _deterministic_forward_compare(av, rl)


if __name__ == "__main__":
    main()
