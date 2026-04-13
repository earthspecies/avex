"""Record or diff `run_evaluate` probe metrics for cross-Python comparisons.

Examples
--------
Record a snapshot for the current interpreter (requires ``esp_data``)::

    uv run python scripts/record_evaluate_end_to_end_metrics.py record

Compare two snapshots produced by ``record``::

    uv run python scripts/record_evaluate_end_to_end_metrics.py diff snap_310.json snap_313.json

Merge ``record`` output into ``tests/fixtures/evaluate_end_to_end_metric_baselines.json``
under ``profiles.<py310_312|py313_plus>`` to enable strict checks in
``tests/integration/test_run_evaluate_cross_version_metrics.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_on_path() -> None:
    root_str = str(_REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def cmd_record() -> int:
    """Print one JSON snapshot from the minimal evaluate harness.

    Returns:
        Process exit code (0 for success).
    """
    _ensure_repo_on_path()
    from tests.integration.eval_end_to_end_harness import (
        python_metrics_profile,
        run_linear_offline_probe_evaluate,
    )

    with tempfile.TemporaryDirectory() as tmp:
        metrics = run_linear_offline_probe_evaluate(Path(tmp))
    payload = {"profile": python_metrics_profile(), "metrics": metrics}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_diff(path_a: Path, path_b: Path) -> int:
    """Print absolute deltas for each metric shared by two snapshot files.

    Args:
        path_a: Path to snapshot JSON A.
        path_b: Path to snapshot JSON B.

    Returns:
        Process exit code (0 for success).
    """
    a: dict[str, Any] = json.loads(path_a.read_text(encoding="utf-8"))
    b: dict[str, Any] = json.loads(path_b.read_text(encoding="utf-8"))
    ma = a.get("metrics") or {}
    mb = b.get("metrics") or {}
    keys = sorted(set(ma) | set(mb))
    print(f"profile_a={a.get('profile')!r} profile_b={b.get('profile')!r}")
    print(f"{'metric':<28} {'a':>12} {'b':>12} {'|a-b|':>12}")
    for k in keys:
        va, vb = ma.get(k), mb.get(k)
        if va is None or vb is None:
            print(f"{k:<28} {repr(va):>12} {repr(vb):>12} {'n/a':>12}")
            continue
        fa, fb = float(va), float(vb)
        print(f"{k:<28} {fa:12.6f} {fb:12.6f} {abs(fa - fb):12.6f}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed CLI args.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("record", help="Run minimal evaluate and print JSON snapshot")

    p_diff = sub.add_parser("diff", help="Compare two JSON snapshots from record")
    p_diff.add_argument("path_a", type=Path)
    p_diff.add_argument("path_b", type=Path)

    return parser.parse_args()


def main() -> int:
    """Entry point.

    Returns:
        Process exit code (0 for success).
    """
    args = parse_args()
    if args.command == "record":
        return cmd_record()
    if args.command == "diff":
        return cmd_diff(args.path_a, args.path_b)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
