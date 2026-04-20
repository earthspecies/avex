"""Upload local markdown model cards to their respective GCS model folders.

This script uploads the contents of:

    evaluation_results/model_cards_markdown/<model_name>/*

to:

    gs://.../models/<model_name>/

It is designed for the common case where each subfolder name corresponds exactly
to a model folder on GCS (after any renames).

By default, this is a dry run (prints the `gsutil` commands). Use `--apply` to
execute uploads.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UploadPlan:
    """A planned upload for one model card folder."""

    model_name: str
    src_dir: Path
    dst_prefix: str
    commands: list[list[str]]


def _print_cmd(argv: list[str]) -> None:
    print(shlex.join(argv))


def _run(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def _gsutil_prefix(base_gcs: str, model_name: str) -> str:
    base = base_gcs.rstrip("/")
    return f"{base}/{model_name}/"


def _iter_model_card_dirs(source_dir: Path) -> list[Path]:
    """Return immediate subdirectories (model card folders) under source_dir.

    Parameters
    ----------
    source_dir : Path
        Root directory that contains per-model card subdirectories.

    Returns
    -------
    list[Path]
        Sorted list of immediate subdirectories representing model card folders.
    """
    out: list[Path] = []
    for child in source_dir.iterdir():
        if child.is_dir():
            out.append(child)
    return sorted(out, key=lambda p: p.name)


def _plan_rsync(
    *,
    gsutil_bin: str,
    parallel: bool,
    src_dir: Path,
    dst_prefix: str,
) -> list[list[str]]:
    # gsutil parallel mode is a global flag before the command: `gsutil -m rsync ...`
    argv: list[str] = [gsutil_bin]
    if parallel:
        argv.append("-m")
    argv.extend(["rsync", "-r", str(src_dir), dst_prefix])
    return [argv]


def _plan_cp_files(
    *,
    gsutil_bin: str,
    parallel: bool,
    src_dir: Path,
    dst_prefix: str,
) -> list[list[str]]:
    # Fallback: upload each file individually, preserving relative paths.
    commands: list[list[str]] = []
    for file_path in sorted(src_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(src_dir).as_posix()
        dst = dst_prefix + rel
        argv: list[str] = [gsutil_bin]
        if parallel:
            argv.append("-m")
        argv.extend(["cp", str(file_path), dst])
        commands.append(argv)
    return commands


def build_plans(
    *,
    source_dir: Path,
    base_gcs: str,
    gsutil_bin: str,
    parallel: bool,
    mode: str,
    only_models: Iterable[str] | None,
) -> list[UploadPlan]:
    """Build upload plans for each model card folder.

    Parameters
    ----------
    source_dir : Path
        Local directory containing per-model markdown folders.
    base_gcs : str
        Base GCS prefix under which model folders live.
    gsutil_bin : str
        Path to the `gsutil` executable.
    parallel : bool
        Whether to enable `gsutil -m` parallel mode.
    mode : str
        Upload mode, either ``\"rsync\"`` or ``\"cp\"``.
    only_models : Iterable[str] | None
        Optional iterable of model names to include; if None, include all.

    Returns
    -------
    list[UploadPlan]
        List of planned uploads, one per model card folder.

    Raises
    ------
    FileNotFoundError
        If ``source_dir`` does not exist.
    NotADirectoryError
        If ``source_dir`` is not a directory.
    ValueError
        If an unknown ``mode`` is provided.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    only_set = set(only_models) if only_models else None

    plans: list[UploadPlan] = []
    for model_dir in _iter_model_card_dirs(source_dir):
        model_name = model_dir.name
        if only_set is not None and model_name not in only_set:
            continue

        dst_prefix = _gsutil_prefix(base_gcs, model_name)
        if mode == "rsync":
            commands = _plan_rsync(
                gsutil_bin=gsutil_bin,
                parallel=parallel,
                src_dir=model_dir,
                dst_prefix=dst_prefix,
            )
        elif mode == "cp":
            commands = _plan_cp_files(
                gsutil_bin=gsutil_bin,
                parallel=parallel,
                src_dir=model_dir,
                dst_prefix=dst_prefix,
            )
        else:
            raise ValueError(f"Unknown mode: {mode!r} (expected 'rsync' or 'cp')")

        plans.append(
            UploadPlan(
                model_name=model_name,
                src_dir=model_dir,
                dst_prefix=dst_prefix,
                commands=commands,
            )
        )

    return plans


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("evaluation_results/model_cards_markdown"),
        help="Local directory containing per-model markdown folders.",
    )
    parser.add_argument(
        "--base-gcs",
        type=str,
        default="gs://representation-learning/esp-avex/models",
        help="Base GCS prefix containing model folders (no trailing slash needed).",
    )
    parser.add_argument(
        "--gsutil",
        type=str,
        default="gsutil",
        help="Path to the gsutil executable (default: gsutil).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable gsutil parallel mode (-m).",
    )
    parser.add_argument(
        "--mode",
        choices=["rsync", "cp"],
        default="rsync",
        help="Upload strategy: rsync (recommended) or cp (fallback).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of model folder names to upload (defaults to all).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute uploads (default: dry-run / print commands only).",
    )

    args = parser.parse_args()

    plans = build_plans(
        source_dir=args.source_dir,
        base_gcs=args.base_gcs,
        gsutil_bin=args.gsutil,
        parallel=not args.no_parallel,
        mode=args.mode,
        only_models=args.only,
    )

    if not plans:
        logger.info("No model card folders found to upload.")
        return

    logger.info("Planned uploads (%d):", len(plans))
    for plan in plans:
        logger.info("  - %s -> %s", plan.src_dir.as_posix(), plan.dst_prefix)

    for plan in plans:
        logger.info("Uploading: %s", plan.model_name)
        for cmd in plan.commands:
            if args.apply:
                _run(cmd)
            else:
                _print_cmd(cmd)


if __name__ == "__main__":
    main()
