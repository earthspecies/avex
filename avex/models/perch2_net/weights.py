"""Resolve Perch v2 PyTorch weights: cached file, or first-use conversion of the pinned ONNX.

Weights are never redistributed. On first use the pinned, checksum-verified
ONNX export is converted (:mod:`avex.models.perch2_net.convert`) into a
versioned safetensors file in the avex cache and reused afterwards.

Resolution order (``RESOLVERS``; an ``hf://`` resolver can be inserted in
front later without touching the model code):

1. an explicit ``weights_path``;
2. the cache file ``<cache>/perch2/perch2-v{FORMAT_VERSION}.safetensors``;
3. conversion from an explicit ``onnx_path``, ``$AVEX_PERCH2_ONNX``, the
   Hugging Face cache, or a download (unless downloads are disabled).

The cache lives in avex's cache root, ``$ESP_CACHE_HOME`` (default
``~/.cache/esp``), under ``perch2/``. Run ``avex fetch perch2`` on a machine
with internet access to populate it before using compute nodes without
network.

Integrity: the input ONNX is pinned by SHA-256. The converted file is checked
structurally on every load (exact keys and shapes, float32, finite) and its
metadata must name the pinned ONNX and the current format version. A fresh
conversion is also checked against reference outputs before it is written.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from avex.models.perch2_net import golden
from avex.models.perch2_net.network import Perch2Config, Perch2Network

logger = logging.getLogger(__name__)

ONNX_REPO_ID = "justinchuby/Perch-onnx"
ONNX_REVISION = "da88bf8c37c0b3068fc706fe509f633b9fbd28c1"
ONNX_FILENAME = "perch_v2.onnx"
ONNX_SHA256 = "bf0c8467a924cb074663970ca4a0ab1e143602121930209657d0dff5d5cefa1f"

# Bump when the conversion or the state-dict layout changes: the version is
# part of the cache filename, so stale caches are ignored, not reused.
FORMAT_VERSION = 1

ONNX_ENV_VAR = "AVEX_PERCH2_ONNX"


class Perch2WeightsError(RuntimeError):
    """Perch v2 weights are unavailable or failed verification."""


def cache_dir() -> Path:
    """Directory holding converted Perch v2 weights.

    Uses the same root as avex's checkpoint cache (``ESP_CACHE_HOME``); an
    empty value is treated as unset.

    Returns
    -------
    Path
        ``$ESP_CACHE_HOME/perch2`` or ``~/.cache/esp/perch2``.
    """
    root = os.environ.get("ESP_CACHE_HOME") or str(Path.home() / ".cache" / "esp")
    return Path(root).expanduser() / "perch2"


def cache_path() -> Path:
    """Path of the converted weights for the current format version.

    Returns
    -------
    Path
        ``<cache_dir>/perch2-v{FORMAT_VERSION}.safetensors``.
    """
    return cache_dir() / f"perch2-v{FORMAT_VERSION}.safetensors"


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file.

    Returns
    -------
    str
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_onnx(path: Path) -> None:
    """Check that ``path`` is the pinned Perch v2 ONNX file.

    Raises
    ------
    Perch2WeightsError
        If the file is missing or its SHA-256 differs from ``ONNX_SHA256``.
    """
    path = Path(path)
    if not path.is_file():
        raise Perch2WeightsError(f"Perch v2 ONNX file not found: {path}")
    actual = sha256_file(path)
    if actual != ONNX_SHA256:
        raise Perch2WeightsError(
            f"{path} is not the pinned Perch v2 ONNX export "
            f"(sha256 {actual}, expected {ONNX_SHA256} = {ONNX_REPO_ID}@{ONNX_REVISION[:8]}/{ONNX_FILENAME})."
        )


def download_onnx(dest_dir: Path) -> Path:
    """Download the pinned ONNX into ``dest_dir`` (bypassing the HF cache).

    Returns
    -------
    Path
        Path of the downloaded file.

    Raises
    ------
    Perch2WeightsError
        If the download fails (e.g. no network access).
    """
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(ONNX_REPO_ID, ONNX_FILENAME, revision=ONNX_REVISION, local_dir=str(dest_dir)))
    except Exception as e:
        raise Perch2WeightsError(
            f"Could not download the Perch v2 ONNX export ({ONNX_REPO_ID}@{ONNX_REVISION[:8]}/{ONNX_FILENAME}): {e}\n"
            f"Converted weights are expected at {cache_path()}. On a machine with internet access run "
            "`avex fetch perch2` (with the same ESP_CACHE_HOME), or download the file yourself and pass "
            f"`onnx_path=` / set ${ONNX_ENV_VAR}."
        ) from e


def onnx_in_hf_cache() -> Path:
    """Download the pinned ONNX into the Hugging Face cache (dev tooling, tests).

    Unlike :func:`download_onnx`, the file stays in the shared HF cache, where
    first-use conversion also looks for it, so it is downloaded at most once.

    Returns
    -------
    Path
        Path of the verified file in the HF cache.
    """
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(ONNX_REPO_ID, ONNX_FILENAME, revision=ONNX_REVISION))
    verify_onnx(path)
    return path


def _hf_cached_onnx() -> Optional[Path]:
    """Return the pinned ONNX if it is already in the Hugging Face cache.

    Returns
    -------
    Optional[Path]
        Cached file path, or None.
    """
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(ONNX_REPO_ID, ONNX_FILENAME, revision=ONNX_REVISION)
    return Path(found) if isinstance(found, str) else None


def verify_state_dict(state: dict[str, torch.Tensor], config: Optional[Perch2Config] = None) -> None:
    """Check keys, shapes, dtype and finiteness of a converted state dict.

    Raises
    ------
    Perch2WeightsError
        On any mismatch.
    """
    from avex.models.perch2_net.convert import expected_shapes

    expected = expected_shapes(config)
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    if missing or unexpected:
        raise Perch2WeightsError(
            f"Perch v2 weights have the wrong keys (missing {sorted(missing)[:5]}, unexpected {sorted(unexpected)[:5]})"
        )
    for key, shape in expected.items():
        tensor = state[key]
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            raise Perch2WeightsError(f"{key}: {tuple(tensor.shape)} {tensor.dtype}, expected {shape} float32")
        if not torch.isfinite(tensor).all():
            raise Perch2WeightsError(f"{key} contains NaN or Inf")


def self_check(state: dict[str, torch.Tensor], config: Optional[Perch2Config] = None) -> None:
    """Run golden clip 0 through a freshly converted model and compare with ONNX Runtime.

    Raises
    ------
    Perch2WeightsError
        If outputs differ from the reference values in the golden module.
    """
    net = Perch2Network(config).eval()
    net.load_state_dict(state, strict=True)
    clip = torch.from_numpy(golden.golden_clips()[:1])
    with torch.no_grad():
        out = net(clip)
    emb, logits = out.embedding[0], out.label[0]
    expected = torch.tensor(golden.SELF_CHECK_EMBEDDING_HEAD)
    problems = []
    if not torch.allclose(emb[: len(expected)], expected, atol=golden.SELF_CHECK_ATOL, rtol=0):
        problems.append(f"embedding[:8] {emb[:8].tolist()} != {expected.tolist()}")
    if abs(float(emb.norm()) - golden.SELF_CHECK_EMBEDDING_NORM) > golden.SELF_CHECK_ATOL * 10:
        problems.append(f"embedding norm {float(emb.norm()):.6f} != {golden.SELF_CHECK_EMBEDDING_NORM}")
    if int(logits.argmax()) != golden.SELF_CHECK_TOP_LABEL:
        problems.append(f"top label {int(logits.argmax())} != {golden.SELF_CHECK_TOP_LABEL}")
    if abs(float(logits.max()) - golden.SELF_CHECK_TOP_LOGIT) > golden.SELF_CHECK_ATOL * 10:
        problems.append(f"top logit {float(logits.max()):.6f} != {golden.SELF_CHECK_TOP_LOGIT}")
    if problems:
        raise Perch2WeightsError("Converted Perch v2 weights failed the self-check: " + "; ".join(problems))


def convert_to_cache(onnx_path: Path, target: Path) -> Path:
    """Convert ``onnx_path`` and atomically write the verified result to ``target``.

    Returns
    -------
    Path
        ``target``.
    """
    from safetensors.torch import save_file

    from avex.models.perch2_net.convert import convert_onnx

    verify_onnx(onnx_path)
    logger.info("Converting Perch v2 ONNX %s -> %s (one-time, needs ~1.5 GB RAM)", onnx_path, target)
    state = {k: torch.from_numpy(v) for k, v in convert_onnx(onnx_path).items()}
    verify_state_dict(state)
    self_check(state)

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".tmp")
    os.close(fd)
    try:
        metadata = {"format_version": str(FORMAT_VERSION), "source_onnx_sha256": ONNX_SHA256}
        save_file(state, tmp, metadata=metadata)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; shared caches need group/other read
        sidecar_tmp = Path(f"{tmp}.sha256")
        sidecar_tmp.write_text(sha256_file(Path(tmp)) + "\n")
        # Weights first: a crash in between leaves a missing sidecar ("unknown"
        # hash in the log), never a sidecar describing another file.
        os.replace(tmp, target)
        os.replace(sidecar_tmp, Path(f"{target}.sha256"))
    finally:
        for leftover in (Path(tmp), Path(f"{tmp}.sha256")):
            leftover.unlink(missing_ok=True)
    return target


@dataclass
class _Request:
    """Inputs shared by the resolvers."""

    weights_path: Optional[Path]
    onnx_path: Optional[Path]
    allow_download: bool


def _from_explicit_path(req: _Request) -> Optional[Path]:
    """Resolver: an explicitly given converted file.

    Returns
    -------
    Optional[Path]
        ``req.weights_path`` or None.

    Raises
    ------
    Perch2WeightsError
        If the given path does not exist.
    """
    if req.weights_path is None:
        return None
    if not req.weights_path.is_file():
        raise Perch2WeightsError(f"Perch v2 weights not found: {req.weights_path}")
    return req.weights_path


def _from_cache(req: _Request) -> Optional[Path]:  # noqa: ARG001
    """Resolver: the versioned cache file.

    Returns
    -------
    Optional[Path]
        The cache file if it exists, else None.
    """
    path = cache_path()
    return path if path.is_file() else None


def _from_conversion(req: _Request) -> Optional[Path]:
    """Resolver: convert the pinned ONNX into the cache, under a file lock.

    Concurrent processes (DDP ranks, DataLoader workers) wait for the lock and
    re-check the cache, so only one converts.

    Returns
    -------
    Optional[Path]
        The cache file.

    Raises
    ------
    Perch2WeightsError
        If no ONNX file is available and downloads are disabled.
    """
    from filelock import FileLock

    target = cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target.parent / "perch2.lock")):
        if target.is_file():  # converted by another process while we waited
            return target
        explicit = req.onnx_path or (Path(os.environ[ONNX_ENV_VAR]) if os.environ.get(ONNX_ENV_VAR) else None)
        source = explicit or _hf_cached_onnx()
        if source is not None:
            return convert_to_cache(source, target)
        if not req.allow_download:
            raise Perch2WeightsError(
                f"Perch v2 weights are not cached at {target} and downloads are disabled. Run `avex fetch perch2` "
                f"on a machine with internet access, or pass `onnx_path=` / set ${ONNX_ENV_VAR}."
            )
        tmp_dir = Path(tempfile.mkdtemp(dir=target.parent, prefix="onnx-download-"))
        try:
            logger.info("Downloading Perch v2 ONNX (%s, ~410 MB)", ONNX_REPO_ID)
            return convert_to_cache(download_onnx(tmp_dir), target)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


RESOLVERS: list[Callable[[_Request], Optional[Path]]] = [_from_explicit_path, _from_cache, _from_conversion]


def resolve_weights(
    *,
    weights_path: Optional[str | Path] = None,
    onnx_path: Optional[str | Path] = None,
    allow_download: bool = True,
) -> Path:
    """Return a path to converted Perch v2 weights, converting on first use.

    Returns
    -------
    Path
        A safetensors file to pass to :func:`load_weights`.

    Raises
    ------
    Perch2WeightsError
        If no resolver produced a file.
    """
    req = _Request(
        Path(weights_path) if weights_path else None,
        Path(onnx_path) if onnx_path else None,
        allow_download,
    )
    for resolver in RESOLVERS:
        path = resolver(req)
        if path is not None:
            return path
    raise Perch2WeightsError("No Perch v2 weights available")


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    """Load and verify a converted Perch v2 safetensors file.

    Returns
    -------
    dict[str, torch.Tensor]
        State dict for :class:`Perch2Network`.

    Raises
    ------
    Perch2WeightsError
        If the metadata names a different format or source, or the tensors
        fail structural verification.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
    if metadata.get("format_version") != str(FORMAT_VERSION):
        raise Perch2WeightsError(
            f"{path} has format version {metadata.get('format_version')!r}, expected {FORMAT_VERSION}. "
            "Delete it to reconvert."
        )
    if metadata.get("source_onnx_sha256") != ONNX_SHA256:
        raise Perch2WeightsError(f"{path} was converted from a different ONNX file. Delete it to reconvert.")
    state = load_file(str(path))
    verify_state_dict(state)
    sidecar = Path(f"{path}.sha256")
    file_hash = sidecar.read_text().strip() if sidecar.is_file() else "unknown"
    logger.info(
        "Loaded Perch v2 weights %s (format v%s, source ONNX sha256 %s, file sha256 %s)",
        path,
        FORMAT_VERSION,
        ONNX_SHA256[:12],
        file_hash[:12],
    )
    return state
