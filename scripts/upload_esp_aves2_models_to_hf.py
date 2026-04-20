"""Upload ESP-AVES2 model folders from GCS to the Hugging Face Hub.

This script:
1) Downloads each model folder from GCS (via `gsutil rsync -r`) to a temp dir
2) Converts .pt checkpoints to .safetensors and writes a SHA-256 hash file
3) Creates/updates `EarthSpeciesProject/<model_name>` model repos on the Hub
4) Uploads all files (safetensors, .safetensors.sha256, metadata/model card)
5) Adds each repo to a Hub Collection named `esp-aves2` (creating it if needed)

Models listed in EXCLUDED_MODELS (e.g. esp-aves2-eat-all-e26, esp-aves2-eat-audioset-e30)
are never uploaded.

Notes
-----
- Requires a Hugging Face token with write access to the `EarthSpeciesProject` org.
  Set it via `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`, or pass `--token`.
- Uses `gsutil` for GCS access; ensure you are authenticated (`gcloud auth login`)
  and have bucket permissions.
- By default, this is a dry run (prints planned actions). Use `--apply` to execute.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Models to skip when uploading (e.g. deprecated, not for public release, or invalid HF repo id).
EXCLUDED_MODELS: frozenset[str] = frozenset(
    {
        "esp-aves2-eat-all-e26",
        "esp-aves2-eat-audioset-e30",
        "esp-aves2-effnetb0-all-",  # HF repo id cannot end with '-'
    }
)

DEFAULT_MODELS: tuple[str, ...] = (
    "esp-aves2-eat-all",
    "esp-aves2-eat-all-e26",
    "esp-aves2-eat-audioset-e30",
    "esp-aves2-eat-bio",
    "esp-aves2-effnetb0-all",
    "esp-aves2-effnetb0-all-",  # trailing '-' comes from trailing '_' in original
    "esp-aves2-effnetb0-audioset",
    "esp-aves2-effnetb0-bio",
    "esp-aves2-naturelm-audio-v1-beats",
    "esp-aves2-sl-beats-all",
    "esp-aves2-sl-beats-bio",
    "esp-aves2-sl-eat-all-ssl-all",
    "esp-aves2-sl-eat-bio-ssl-all",
)


@dataclass(frozen=True, slots=True)
class ModelPlan:
    """Plan for one model upload."""

    model_name: str
    gcs_prefix: str
    hf_repo_id: str


def _repo_root() -> Path:
    # scripts/<this_file>.py -> repo root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def _print_cmd(argv: list[str]) -> None:
    print(shlex.join(argv))


def _run(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def _gcs_prefix(base_gcs: str, model_name: str) -> str:
    base = base_gcs.rstrip("/")
    return f"{base}/{model_name}/"


def _resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token

    # Fall back to the local Hugging Face token store (e.g. after `hf auth login`).
    try:
        from huggingface_hub import HfFolder  # type: ignore

        return HfFolder.get_token()
    except Exception:  # noqa: BLE001 - best-effort fallback
        return None


def _ensure_hf_collection(
    *,
    namespace: str,
    title: str,
    private: bool,
    token: str,
) -> str:
    """Get or create a collection and return its `collection_slug`.

    Parameters
    ----------
    namespace : str
        Owner namespace (user or organization) of the collection.
    title : str
        Human-readable collection title.
    private : bool
        Whether the collection should be private.
    token : str
        Hugging Face authentication token.

    Returns
    -------
    str
        The Hugging Face collection slug, for example `org/title-<id>`.

    Raises
    ------
    RuntimeError
        If `huggingface_hub` is unavailable or does not support collections.
    """
    try:
        from huggingface_hub import create_collection, list_collections  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required (and must support collections). Install/upgrade `huggingface_hub`."
        ) from e

    # Find existing collection by title under the owner (org/user).
    for col in list_collections(owner=namespace, token=token):
        if getattr(col, "title", None) == title:
            return col.slug

    created = create_collection(
        title=title,
        description="ESP-AVES2 model zoo.",
        namespace=namespace,
        private=private,
        exists_ok=True,
        token=token,
    )
    return created.slug


def _add_repo_to_collection(
    *,
    collection_slug: str,
    repo_id: str,
    token: str,
) -> None:
    try:
        from huggingface_hub import add_collection_item  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required (and must support collections). Install/upgrade `huggingface_hub`."
        ) from e

    add_collection_item(
        collection_slug=collection_slug,
        item_id=repo_id,
        item_type="model",
        note="Uploaded from GCS.",
        exists_ok=True,
        token=token,
    )


def _download_model_folder(
    *,
    gsutil_bin: str,
    parallel: bool,
    gcs_prefix: str,
    local_dir: Path,
    dry_run: bool,
) -> None:
    argv: list[str] = [gsutil_bin]
    if parallel:
        argv.append("-m")
    argv.extend(["rsync", "-r", gcs_prefix, str(local_dir)])
    if dry_run:
        _print_cmd(argv)
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    _run(argv)


def _find_pt_checkpoint(local_dir: Path, model_name: str, *, dry_run: bool) -> Path:
    preferred = local_dir / f"{model_name}.pt"
    if dry_run:
        return preferred
    if preferred.is_file():
        return preferred
    # Fallback: if the model name has no matching file (e.g. renamed folder),
    # pick the only .pt in the directory.
    pts = sorted(p for p in local_dir.glob("*.pt") if p.is_file())
    if len(pts) == 1:
        return pts[0]
    raise FileNotFoundError(
        f"Could not find checkpoint .pt for {model_name}. Expected {preferred.name} or exactly one *.pt in {local_dir}."
    )


def _convert_pt_to_safetensors(
    *,
    checkpoint_pt: Path,
    verify: str,
    dry_run: bool,
) -> Path:
    """Convert a local .pt checkpoint to .safetensors using the existing script.

    Parameters
    ----------
    checkpoint_pt : Path
        Path to the source `.pt` checkpoint.
    verify : str
        Verification mode to pass through to the conversion script.
    dry_run : bool
        If True, print the command and return the expected destination path
        without running the conversion.

    Returns
    -------
    Path
        Path to the generated `.safetensors` file.

    Raises
    ------
    FileNotFoundError
        If the conversion script is missing or the expected output file cannot
        be located after conversion.
    """
    convert_script = _repo_root() / "scripts" / "convert_to_safetensors.py"
    if not convert_script.is_file():
        raise FileNotFoundError(f"Missing conversion script: {convert_script}")

    argv = [
        sys.executable,
        str(convert_script),
        str(checkpoint_pt),
        "--verify",
        verify,
        "--no-manifest",
    ]
    if dry_run:
        _print_cmd(argv)
        # Best-effort: return the expected location (script nests it in a folder).
        stem = checkpoint_pt.stem
        return checkpoint_pt.parent / stem / f"{stem}.safetensors"

    _run(argv)

    # Conversion script outputs `<dir>/<stem>/<stem>.safetensors` for local inputs.
    stem = checkpoint_pt.stem
    out = checkpoint_pt.parent / stem / f"{stem}.safetensors"
    if not out.is_file():
        # Fallback: search if behavior changes.
        candidates = sorted(checkpoint_pt.parent.rglob("*.safetensors"))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(f"Expected converted file at {out}, found: {candidates}")
    return out


def _load_model_spec_from_train_config(train_config_path: Path) -> dict:
    import yaml

    data = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "model_spec" not in data:
        raise ValueError(f"train_config.yaml missing model_spec: {train_config_path}")
    model_spec = data["model_spec"]
    if not isinstance(model_spec, dict):
        raise ValueError(f"train_config.yaml model_spec must be a mapping: {train_config_path}")
    return model_spec


def _api_load_model_sanity(
    *,
    train_config_path: Path,
    checkpoint_path: Path,
    dry_run: bool,
) -> None:
    """Sanity-check that `load_model` can construct and load weights.

    Uses `model_spec` from `train_config.yaml` and loads on CPU.
    """
    # This check is optional and can be slow (it may download pretrained backbone
    # weights from third-party sources). Gate it behind an env flag so that
    # publishing runs stay focused on conversion + upload.
    if os.environ.get("AVEX_UPLOAD_SANITY", "").strip().lower() not in {"1", "true", "yes"}:
        logger.info("Skipping API sanity-load (set AVEX_UPLOAD_SANITY=1 to enable).")
        return

    if dry_run:
        print(f"[DRY RUN] Would sanity-load via API: {checkpoint_path}")
        return

    # This script originally lived in a development branch where the package was
    # named `representation_learning`. In `avex`, the equivalent public API lives
    # under `avex.*`. If neither import is available (e.g. minimal env), treat
    # this step as optional and continue.
    try:
        from avex.configs import ModelSpec  # type: ignore[attr-defined]
        from avex.models.utils.load import load_model  # type: ignore
    except Exception:  # noqa: BLE001 - optional best-effort sanity check
        logger.warning(
            "Skipping API sanity-load: could not import avex ModelSpec/load_model. "
            "This does not block conversion or upload."
        )
        return

    model_spec_dict = _load_model_spec_from_train_config(train_config_path)

    # Ensure this check is CPU-only and doesn't depend on external pretrained weights paths.
    model_spec_dict = dict(model_spec_dict)
    model_spec_dict["device"] = "cpu"
    model_spec_dict["pretrained"] = False
    if "fairseq_weights_path" in model_spec_dict:
        model_spec_dict["fairseq_weights_path"] = None

    spec = ModelSpec.model_validate(model_spec_dict)
    _ = load_model(spec, device="cpu", checkpoint_path=str(checkpoint_path))


def _promote_safetensors_to_root(
    *,
    local_dir: Path,
    safetensors_path: Path,
    model_name: str,
    dry_run: bool,
) -> Path:
    """Move the converted safetensors file to `<local_dir>/<model_name>.safetensors`.

    The conversion script nests outputs under `<local_dir>/<stem>/...`. For a clean
    Hub repo layout, we promote the weights to the model folder root.

    Parameters
    ----------
    local_dir : Path
        Local directory corresponding to the model repository.
    safetensors_path : Path
        Path to the generated `.safetensors` file.
    model_name : str
        Name of the model, used for the destination filename.
    dry_run : bool
        If True, log the planned move and return the destination without
        performing file operations.

    Returns
    -------
    Path
        Destination path `<local_dir>/<model_name>.safetensors`.
    """
    dst = local_dir / f"{model_name}.safetensors"
    if dry_run:
        print(f"[DRY RUN] Would move {safetensors_path} -> {dst}")
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    if safetensors_path.resolve() == dst.resolve():
        return dst
    dst.write_bytes(safetensors_path.read_bytes())
    return dst


def _cleanup_nested_conversion_dir(*, local_dir: Path, checkpoint_stem: str, dry_run: bool) -> None:
    """Remove `<local_dir>/<checkpoint_stem>/` produced by conversion (best effort)."""
    nested = local_dir / checkpoint_stem
    if dry_run:
        print(f"[DRY RUN] Would remove conversion dir (optional): {nested}")
        return
    if not nested.is_dir():
        return
    # Best effort cleanup; ignore failures to avoid masking upload progress.
    try:
        for p in sorted(nested.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        nested.rmdir()
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not clean up %s: %s", nested, e)


_HF_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _sanitize_hf_readme_front_matter(*, local_dir: Path, dry_run: bool) -> None:
    """Sanitize README.md front matter to satisfy HF Hub validation.

    The Hub validates YAML metadata in README.md. In particular, `base_model` must
    contain Hub model IDs (like `org/model-name`), not free-form strings. Our
    generated model cards sometimes use descriptive strings.
    """
    readme = local_dir / "README.md"
    if not readme.is_file():
        return

    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return

    parts = text.split("---", 2)
    if len(parts) < 3:
        return
    _before, yaml_block, rest = parts[0], parts[1], parts[2]

    try:
        import yaml

        meta = yaml.safe_load(yaml_block) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not parse README.md front matter; leaving as-is (%s): %s", readme, e)
        return

    if not isinstance(meta, dict):
        return

    base_model = meta.get("base_model")
    if isinstance(base_model, list):
        # Drop base_model entirely if any entry is not a valid HF model id.
        if any(not isinstance(x, str) or _HF_MODEL_ID_RE.fullmatch(x) is None for x in base_model):
            logger.info("Dropping invalid README.md metadata field: base_model")
            meta.pop("base_model", None)
    elif base_model is not None:
        # Non-list base_model is invalid for our use; drop it.
        logger.info("Dropping invalid README.md metadata field: base_model")
        meta.pop("base_model", None)

    new_yaml = yaml.safe_dump(meta, sort_keys=False).strip()
    new_text = f"---\n{new_yaml}\n---{rest}"

    if dry_run:
        print(f"[DRY RUN] Would sanitize README.md front matter: {readme}")
        return

    readme.write_text(new_text, encoding="utf-8")


def _upload_folder_to_hf(
    *,
    repo_id: str,
    local_dir: Path,
    token: str,
    dry_run: bool,
    commit_message: str,
    include_pt: bool,
    private: bool,
) -> None:
    if dry_run:
        print(f"[DRY RUN] Would create/update HF repo: {repo_id}")
        print(f"[DRY RUN] Would upload folder: {local_dir} -> {repo_id}")
        return

    from huggingface_hub import HfApi, create_repo  # type: ignore

    create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=private,
        token=token,
    )
    api = HfApi(token=token)
    ignore_patterns = None if include_pt else ["*.pt", "*.pth"]
    _sanitize_hf_readme_front_matter(local_dir=local_dir, dry_run=False)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        path_in_repo=".",
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
    )


def _write_safetensors_sha256(
    *,
    safetensors_path: Path,
    model_name: str,
    dry_run: bool,
) -> Path:
    """Write a SHA-256 hash file for the safetensors next to the weights file.

    Creates `<model_name>.safetensors.sha256` with content:
        <hex_hash>  <model_name>.safetensors

    so it is compatible with `sha256sum -c`.

    Parameters
    ----------
    safetensors_path : Path
        Path to the .safetensors file.
    model_name : str
        Model name; used for the hash file name and the filename in the content.
    dry_run : bool
        If True, log and return the expected hash path without writing.

    Returns
    -------
    Path
        Path to the written (or would-be) .sha256 file.
    """
    sha256_path = safetensors_path.parent / f"{model_name}.safetensors.sha256"
    if dry_run:
        print(f"[DRY RUN] Would write SHA-256 hash: {sha256_path}")
        return sha256_path

    h = hashlib.sha256()
    with open(safetensors_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    hex_digest = h.hexdigest()
    line = f"{hex_digest}  {model_name}.safetensors\n"
    sha256_path.write_text(line, encoding="utf-8")
    logger.info("Wrote %s", sha256_path.name)
    return sha256_path


def build_plans(
    *,
    gcs_base: str,
    hf_namespace: str,
    models: list[str] | None,
) -> list[ModelPlan]:
    chosen = DEFAULT_MODELS if not models else tuple(models)
    out: list[ModelPlan] = []
    for model_name in chosen:
        if model_name in EXCLUDED_MODELS:
            logger.info("Skipping excluded model: %s", model_name)
            continue
        gcs_prefix = _gcs_prefix(gcs_base, model_name)
        hf_repo_id = f"{hf_namespace}/{model_name}"
        out.append(ModelPlan(model_name=model_name, gcs_prefix=gcs_prefix, hf_repo_id=hf_repo_id))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gcs-base",
        type=str,
        default="gs://representation-learning/esp-avex/models",
        help="Base GCS prefix containing model folders.",
    )
    parser.add_argument(
        "--gsutil",
        type=str,
        default="gsutil",
        help="Path to the gsutil executable.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable `gsutil -m`.",
    )
    parser.add_argument(
        "--hf-namespace",
        type=str,
        default="EarthSpeciesProject",
        help="Hugging Face org/user namespace.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update model repos as private.",
    )
    parser.add_argument(
        "--private-collection",
        action="store_true",
        help="Create the collection as private (if it must be created).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="esp-aves2",
        help="Hugging Face collection title to add models to.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model folder names to upload (defaults to ESP-AVES2 list).",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep the temporary download directory (default: delete).",
    )
    parser.add_argument(
        "--apply-local",
        action="store_true",
        help=(
            "Execute local steps (download from GCS, convert to safetensors, "
            "sanity-load via API) but do NOT upload to Hugging Face. "
            "Use --apply for full upload."
        ),
    )
    parser.add_argument(
        "--verify",
        choices=["none", "fast", "full"],
        default="fast",
        help="Verification mode for safetensors conversion (default: fast).",
    )
    parser.add_argument(
        "--include-pt",
        action="store_true",
        help="Upload original .pt files to Hugging Face as well (default: upload only safetensors).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute download + upload (default: dry-run).",
    )
    args = parser.parse_args()

    token = _resolve_token(args.token)
    if args.apply and not token:
        raise SystemExit("Missing Hugging Face token. Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or pass --token.")

    apply_local = bool(args.apply or args.apply_local)
    dry_run = not apply_local
    plans = build_plans(gcs_base=args.gcs_base, hf_namespace=args.hf_namespace, models=args.models)

    logger.info("Planned models (%d):", len(plans))
    for p in plans:
        logger.info("  - %s: %s -> %s", p.model_name, p.gcs_prefix, p.hf_repo_id)

    collection_slug: str | None = None
    if args.apply:
        assert token is not None
        collection_slug = _ensure_hf_collection(
            namespace=args.hf_namespace,
            title=args.collection,
            private=bool(args.private_collection),
            token=token,
        )
        logger.info("Using collection: %s", collection_slug)
    else:
        print(f"[DRY RUN] Would ensure collection exists: {args.hf_namespace}/{args.collection}")

    tmp_root_obj = tempfile.TemporaryDirectory(prefix="esp_aves2_hf_upload_")
    tmp_root = Path(tmp_root_obj.name)
    try:
        for p in plans:
            local_dir = tmp_root / p.model_name
            logger.info("Downloading: %s", p.gcs_prefix)
            _download_model_folder(
                gsutil_bin=args.gsutil,
                parallel=not args.no_parallel,
                gcs_prefix=p.gcs_prefix,
                local_dir=local_dir,
                dry_run=dry_run,
            )

            # Convert to safetensors and verify using the canonical script.
            train_config_path = local_dir / "train_config.yaml"
            checkpoint_pt = _find_pt_checkpoint(local_dir, p.model_name, dry_run=dry_run)
            logger.info("Converting checkpoint to safetensors: %s", checkpoint_pt.name)
            safetensors_path = _convert_pt_to_safetensors(
                checkpoint_pt=checkpoint_pt,
                verify=args.verify,
                dry_run=dry_run,
            )
            safetensors_path = _promote_safetensors_to_root(
                local_dir=local_dir,
                safetensors_path=safetensors_path,
                model_name=p.model_name,
                dry_run=dry_run,
            )
            _cleanup_nested_conversion_dir(
                local_dir=local_dir,
                checkpoint_stem=checkpoint_pt.stem,
                dry_run=dry_run,
            )

            # Create SHA-256 hash file for the safetensors (uploaded with the folder).
            _write_safetensors_sha256(
                safetensors_path=safetensors_path,
                model_name=p.model_name,
                dry_run=dry_run,
            )

            # Ensure API load_model can build and load from the converted safetensors
            # when train_config.yaml is present (skip for legacy folders that lack it).
            if train_config_path.is_file():
                logger.info("Sanity-loading via API from: %s", safetensors_path.name)
                _api_load_model_sanity(
                    train_config_path=train_config_path,
                    checkpoint_path=safetensors_path,
                    dry_run=dry_run,
                )
            else:
                logger.warning(
                    "No train_config.yaml in %s; skipping API sanity-load (upload will continue)",
                    local_dir,
                )

            if args.apply:
                logger.info("Uploading to HF: %s", p.hf_repo_id)
                _upload_folder_to_hf(
                    repo_id=p.hf_repo_id,
                    local_dir=local_dir,
                    token=token or "",
                    dry_run=False,
                    commit_message=f"Upload {p.model_name} from GCS",
                    include_pt=bool(args.include_pt),
                    private=bool(args.private),
                )
            else:
                # Local apply mode (or pure dry-run): do not upload.
                print(f"[DRY RUN] Would create/update HF repo: {p.hf_repo_id}")
                print(f"[DRY RUN] Would upload folder (excluding .pt by default): {local_dir} -> {p.hf_repo_id}")

            if args.apply:
                assert collection_slug is not None
                assert token is not None
                _add_repo_to_collection(
                    collection_slug=collection_slug,
                    repo_id=p.hf_repo_id,
                    token=token,
                )
            else:
                print(f"[DRY RUN] Would add to collection: {p.hf_repo_id} -> {args.collection}")
    finally:
        if args.keep_tmp:
            logger.info("Keeping temp directory: %s", tmp_root)
            tmp_root_obj.cleanup = lambda: None  # type: ignore[assignment]
        else:
            tmp_root_obj.cleanup()


if __name__ == "__main__":
    main()
