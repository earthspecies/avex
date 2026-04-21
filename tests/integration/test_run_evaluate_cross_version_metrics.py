"""Cross-Python tracking for `run_evaluate` probe metrics (internal esp_data).

The minimal end-to-end job in `eval_end_to_end_harness` can yield slightly
different ``test_accuracy`` / ``test_balanced_accuracy`` when PyTorch, NumPy,
or TensorFlow versions differ. This module:

- Always logs a one-line JSON snapshot of the current interpreter's metrics
  (grep CI logs for ``AVEX_EVAL_METRICS_SNAPSHOT``).
- Optionally asserts closeness to values in
  ``tests/fixtures/evaluate_end_to_end_metric_baselines.json`` when the active
  :func:`python_metrics_profile` has a filled entry (if the entry is missing,
  the test still passes after logging the snapshot).

Record baselines per band::

    uv run python scripts/record_evaluate_end_to_end_metrics.py record

Compare two recorded snapshots (e.g. from Python 3.10 vs 3.13 venvs)::

    uv run python scripts/record_evaluate_end_to_end_metrics.py diff a.json b.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("esp_data")

from tests.integration.eval_end_to_end_harness import (
    EVAL_SUMMARY_METRIC_KEYS,
    python_metrics_profile,
    run_linear_offline_probe_evaluate,
)


def _baseline_path() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "evaluate_end_to_end_metric_baselines.json"


def _load_baseline_file() -> dict[str, Any]:
    path = _baseline_path()
    if not path.is_file():
        return {"profiles": {}, "tolerances": {"atol": 0.12, "rtol": 0.05}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.slow
def test_evaluate_end_to_end_probe_metrics_snapshot_and_optional_baseline(
    tmp_path: Path,
) -> None:
    """Run minimal evaluate, log metrics, and compare to baseline if configured."""
    metrics = run_linear_offline_probe_evaluate(tmp_path)

    for name in EVAL_SUMMARY_METRIC_KEYS:
        assert name in metrics, f"Missing metric {name!r} in evaluate summary"
        val = metrics[name]
        assert 0.0 <= val <= 1.0, f"{name} out of valid range: {val}"

    profile = python_metrics_profile()
    snap = json.dumps({"profile": profile, "metrics": metrics}, sort_keys=True)
    print(f"\nAVEX_EVAL_METRICS_SNAPSHOT {snap}\n")

    data = _load_baseline_file()
    tolerances = data.get("tolerances", {})
    atol = float(tolerances.get("atol", 0.12))
    rtol = float(tolerances.get("rtol", 0.05))
    profiles: dict[str, Any] = data.get("profiles") or {}
    expected = profiles.get(profile)

    if expected is None or expected == {}:
        # Baselines are optional: still exercise the pipeline and emit the snapshot for CI logs.
        return

    for key in EVAL_SUMMARY_METRIC_KEYS:
        if key not in expected:
            pytest.fail(
                f"Baseline for profile {profile!r} missing key {key!r}. Expected keys: {list(EVAL_SUMMARY_METRIC_KEYS)}"
            )
        exp_v = float(expected[key])
        got_v = metrics[key]
        if not math.isclose(got_v, exp_v, rel_tol=rtol, abs_tol=atol):
            pytest.fail(
                f"Metric {key!r} drift for profile {profile!r}: "
                f"expected {exp_v} (± atol={atol}, rtol={rtol}), got {got_v}. "
                "Update the baseline JSON if the stack changed intentionally."
            )
