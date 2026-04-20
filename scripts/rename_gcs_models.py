"""Rename model folders/files on GCS by applying a consistent naming transform.

This script is intended for a common layout where each "model folder" contains a
model artifact whose basename matches the folder name, e.g.:

    gs://.../models/esp_aves2_eat_all/
      - esp_aves2_eat_all.safetensors

Requested renames:
- Replace underscores with dashes
- Replace "efficientnet" with "effnetb0"

Because GCS does not support atomic renames, this script uses `gsutil mv`:
1) Rename the model file(s) within the folder (matches `<folder>.*`)
2) Move the entire folder prefix to the new folder name

By default this runs in dry-run mode (prints commands). Use `--apply` to execute.

Examples
--------
Dry run (recommended first):
    python scripts/rename_gcs_models.py

Apply:
    python scripts/rename_gcs_models.py --apply

Custom paths:
    python scripts/rename_gcs_models.py --paths gs://bucket/a/b/old_name/
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GcsFolderUri:
    """A parsed GCS folder URI."""

    bucket: str
    parent_path: str
    folder_name: str

    @property
    def folder_uri(self) -> str:
        """Return `gs://.../parent/folder_name/`."""
        if self.parent_path:
            return f"gs://{self.bucket}/{self.parent_path}/{self.folder_name}/"
        return f"gs://{self.bucket}/{self.folder_name}/"

    @property
    def parent_uri(self) -> str:
        """Return `gs://.../parent/` (or `gs://bucket/` if empty)."""
        if self.parent_path:
            return f"gs://{self.bucket}/{self.parent_path}/"
        return f"gs://{self.bucket}/"


def transform_model_name(name: str) -> str:
    """Apply naming rules: efficientnet→effnetb0 and underscores→dashes.

    Parameters
    ----------
    name : str
        Original model or folder name.

    Returns
    -------
    str
        Transformed name after applying the renaming rules.
    """
    return name.replace("efficientnet", "effnetb0").replace("_", "-")


def parse_gcs_folder_uri(uri: str) -> GcsFolderUri:
    """Parse a `gs://.../folder/` URI into bucket, parent prefix, and folder name.

    Parameters
    ----------
    uri : str
        A GCS URI that points to a folder or prefix.

    Returns
    -------
    GcsFolderUri
        Parsed representation containing bucket, parent path, and folder name.

    Raises
    ------
    ValueError
        If the URI is malformed or missing required components.
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got: {uri!r}")

    rest = uri[len("gs://") :].strip("/")
    if not rest:
        raise ValueError(f"Expected gs://<bucket>/<path>/, got: {uri!r}")

    parts = rest.split("/")
    bucket = parts[0]
    path_parts = parts[1:]
    if not bucket or not path_parts:
        raise ValueError(f"Expected gs://<bucket>/<path>/, got: {uri!r}")

    folder_name = path_parts[-1]
    parent_path = "/".join(path_parts[:-1])
    return GcsFolderUri(bucket=bucket, parent_path=parent_path, folder_name=folder_name)


DEFAULT_PATHS: tuple[str, ...] = (
    "gs://representation-learning/esp-avex/models/esp_aves2_eat_all/",
    "gs://representation-learning/esp-avex/models/esp_aves2_eat_all_e26/",
    "gs://representation-learning/esp-avex/models/esp_aves2_eat_audioset_e30/",
    "gs://representation-learning/esp-avex/models/esp_aves2_eat_bio/",
    "gs://representation-learning/esp-avex/models/esp_aves2_efficientnet_all/",
    "gs://representation-learning/esp-avex/models/esp_aves2_efficientnet_all_/",
    "gs://representation-learning/esp-avex/models/esp_aves2_efficientnet_audioset/",
    "gs://representation-learning/esp-avex/models/esp_aves2_efficientnet_bio/",
    "gs://representation-learning/esp-avex/models/esp_aves2_naturelm_audio_v1_beats/",
    "gs://representation-learning/esp-avex/models/esp_aves2_sl_beats_all/",
    "gs://representation-learning/esp-avex/models/esp_aves2_sl_beats_bio/",
    "gs://representation-learning/esp-avex/models/esp_aves2_sl_eat_all_ssl_all/",
    "gs://representation-learning/esp-avex/models/esp_aves2_sl_eat_bio_ssl_all/",
)


def _run_gsutil(
    gsutil_bin: str,
    args: list[str],
    *,
    capture_output: bool,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    """Run `gsutil` and return the completed process.

    Parameters
    ----------
    gsutil_bin : str
        Path to the `gsutil` executable.
    args : list[str]
        Positional arguments to pass to `gsutil`.
    capture_output : bool
        Whether to capture stdout and stderr.
    check : bool
        Whether to raise an error if the command exits with a non-zero status.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed process with captured output and return code.
    """
    cmd = [gsutil_bin, *args]
    logger.debug("Running: %s", shlex.join(cmd))
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def _gsutil_ls_safe(gsutil_bin: str, pattern: str) -> list[str]:
    """List objects using `gsutil ls`, returning zero lines if none match.

    Parameters
    ----------
    gsutil_bin : str
        Path to the `gsutil` executable.
    pattern : str
        GCS URI pattern to list, such as `gs://bucket/prefix/*`.

    Returns
    -------
    list[str]
        List of matching GCS URIs. Returns an empty list if no objects match.
    """
    # `gsutil ls` exits non-zero when nothing matches. Treat that as "no matches".
    proc = _run_gsutil(gsutil_bin, ["ls", pattern], capture_output=True, check=False)
    if proc.returncode != 0:
        # Surface stderr in case this is auth/permission rather than "no match".
        err = (proc.stderr or "").strip()
        if err:
            logger.warning("gsutil ls failed for %s: %s", pattern, err)
        return []
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return lines


def _print_cmd(cmd: list[str]) -> None:
    print(shlex.join(cmd))


def _mv_flags(*, parallel: bool, no_clobber: bool) -> list[str]:
    flags: list[str] = []
    if parallel:
        flags.append("-m")
    if no_clobber:
        # gsutil mv does not support -n. We implement "no clobber" by checking
        # destination existence before running the mv.
        pass
    return flags


def _gsutil_stat_exists(gsutil_bin: str, uri: str) -> bool:
    """Return True if `uri` exists according to `gsutil stat`.

    Parameters
    ----------
    gsutil_bin : str
        Path to the `gsutil` executable.
    uri : str
        GCS URI to test for existence.

    Returns
    -------
    bool
        True if the object exists according to `gsutil stat`, otherwise False.
    """
    proc = _run_gsutil(gsutil_bin, ["-q", "stat", uri], capture_output=True, check=False)
    return proc.returncode == 0


def _gcs_prefix_has_any_objects(gsutil_bin: str, prefix_uri: str) -> bool:
    """Return True if the prefix contains at least one object.

    Parameters
    ----------
    gsutil_bin : str
        Path to the `gsutil` executable.
    prefix_uri : str
        GCS URI prefix to test, typically ending with a slash.

    Returns
    -------
    bool
        True if any objects exist under the prefix, otherwise False.
    """
    matches = _gsutil_ls_safe(gsutil_bin, f"{prefix_uri}**")
    return len(matches) > 0


def plan_renames(
    *,
    gsutil_bin: str,
    folder_uris: Iterable[str],
    model_exts: Iterable[str],
) -> list[tuple[str, list[list[str]]]]:
    """Plan `gsutil` commands for each folder URI.

    Parameters
    ----------
    gsutil_bin : str
        Path to the `gsutil` executable.
    folder_uris : Iterable[str]
        Iterable of GCS folder URIs to process.
    model_exts : Iterable[str]
        Iterable of model filename extensions to rename (for example, `.pt`).

    Returns
    -------
    list[tuple[str, list[list[str]]]]
        List of `(label, commands)` tuples where each command is argv-style.
    """
    plans: list[tuple[str, list[list[str]]]] = []

    for folder_uri in folder_uris:
        parsed = parse_gcs_folder_uri(folder_uri)
        old_name = parsed.folder_name
        new_name = transform_model_name(old_name)

        if new_name == old_name:
            logger.warning("No-op transform for %s (folder name %r). Skipping.", folder_uri, old_name)
            continue

        old_prefix = parsed.folder_uri  # ends with /
        new_prefix = parsed.parent_uri + new_name + "/"

        # Find model artifacts that match the folder name.
        # We intentionally only rename files at the folder root:
        # `<prefix><old_name><ext>` (e.g., `.pt`).
        model_candidates: list[str] = []
        for ext in model_exts:
            model_candidates.extend(_gsutil_ls_safe(gsutil_bin, f"{old_prefix}{old_name}{ext}"))

        cmds: list[list[str]] = []

        for src in model_candidates:
            # src is a full gs://... URI; keep extension part after the old_name.
            # Example: gs://.../old/old.pt -> ext is ".pt".
            base = src.rsplit("/", 1)[-1]
            suffix = base[len(old_name) :]
            if not suffix.startswith("."):
                # Defensive: only rename things that look like exact basename matches.
                logger.warning("Skipping unexpected candidate (not %s.*): %s", old_name, src)
                continue
            dst = f"{old_prefix}{new_name}{suffix}"
            cmds.append(["gsutil", "mv", src, dst])

        # Move entire folder/prefix (all objects) to the new prefix.
        # Note: with GCS there is no real directory, just prefixes.
        cmds.append(["gsutil", "mv", f"{old_prefix}**", new_prefix])

        label = f"{old_prefix} -> {new_prefix}"
        plans.append((label, cmds))

        if not model_candidates:
            logger.warning(
                "No model file matches found for %s (looked for %s). Folder move will still be planned.",
                old_prefix,
                " or ".join(f"{old_prefix}{old_name}{ext}" for ext in model_exts),
            )

    return plans


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the gsutil commands (default: dry-run / print only).",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional list of gs://.../folder/ URIs to process. Defaults to the built-in list.",
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
        help="Disable gsutil -m for faster transfers (default: parallel enabled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing objects (default: no-clobber).",
    )
    parser.add_argument(
        "--skip-list",
        action="store_true",
        help="Do not list `<folder>.*` model files; only move folder prefixes.",
    )
    parser.add_argument(
        "--model-ext",
        action="append",
        default=None,
        help=("Model filename extension(s) to rename (repeatable). Default: .pt"),
    )
    args = parser.parse_args()

    gsutil_bin: str = args.gsutil
    parallel: bool = not args.no_parallel
    no_clobber: bool = not args.overwrite

    folder_uris = DEFAULT_PATHS if args.paths is None else tuple(args.paths)
    if not folder_uris:
        logger.error("No paths provided.")
        sys.exit(2)

    model_exts = (".pt",) if not args.model_ext else tuple(args.model_ext)

    # If the user requested skip-list, temporarily monkeypatch the safe-ls function
    # to always return no candidates. This keeps planning logic centralized.
    global _gsutil_ls_safe  # noqa: PLW0603 - deliberate internal override
    if args.skip_list:

        def _gsutil_ls_safe(_bin: str, _pattern: str) -> list[str]:  # type: ignore[no-redef]
            """Return an empty list to disable listing when --skip-list is used.

            Parameters
            ----------
            _bin : str
                Unused path to the `gsutil` executable.
            _pattern : str
                Unused GCS URI pattern.

            Returns
            -------
            list[str]
                Always an empty list, indicating no matches.
            """

            return []

    plans = plan_renames(gsutil_bin=gsutil_bin, folder_uris=folder_uris, model_exts=model_exts)
    if not plans:
        logger.info("No renames to perform.")
        return

    # Print a summary first (high signal).
    logger.info("Planned renames (%d):", len(plans))
    for label, _cmds in plans:
        logger.info("  - %s", label)

    # Execute or print.
    for label, cmds in plans:
        logger.info("Processing: %s", label)

        for cmd in cmds:
            # Replace placeholder "gsutil" with the configured binary and add flags.
            if cmd[:2] == ["gsutil", "mv"]:
                mv = [gsutil_bin, *_mv_flags(parallel=parallel, no_clobber=no_clobber), "mv", *cmd[2:]]
                if args.apply:
                    src = cmd[2]
                    dst = cmd[3]

                    # Implement "no clobber" behavior since gsutil mv has no -n.
                    # - For single-object moves: skip if destination exists.
                    # - For prefix moves: skip if destination prefix already has objects.
                    is_prefix_move = src.endswith("**") and dst.endswith("/")
                    if no_clobber:
                        if is_prefix_move:
                            if _gcs_prefix_has_any_objects(gsutil_bin, dst):
                                logger.warning(
                                    "Skipping move because destination prefix is not empty "
                                    "(use --overwrite to force): %s",
                                    dst,
                                )
                                continue
                        else:
                            if _gsutil_stat_exists(gsutil_bin, dst):
                                logger.warning(
                                    "Skipping move because destination exists (use --overwrite to force): %s",
                                    dst,
                                )
                                continue

                    # For prefix moves, treat "matched no objects" as a no-op rather than a hard failure.
                    proc = _run_gsutil(gsutil_bin, mv[1:], capture_output=True, check=False)
                    if proc.returncode != 0:
                        stderr = (proc.stderr or "").strip()
                        if is_prefix_move and "matched no objects" in stderr.lower():
                            logger.info("No objects matched for %s (skipping).", src)
                            continue
                        raise subprocess.CalledProcessError(
                            proc.returncode,
                            proc.args,
                            output=proc.stdout,
                            stderr=proc.stderr,
                        )
                else:
                    _print_cmd(mv)
            else:
                # Should not happen, but keep it explicit.
                if args.apply:
                    _run_gsutil(gsutil_bin, cmd[1:], capture_output=False, check=True)
                else:
                    _print_cmd(cmd)


if __name__ == "__main__":
    main()
