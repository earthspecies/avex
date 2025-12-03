"""Filesystem utilities for local and cloud paths.

This module provides the subset of the ``esp_data.io.filesystem`` interface that is
required by the public API:

- ``filesystem_from_path``: map a path to an fsspec filesystem

The higher-level factory ``filesystem(protocol=...)`` is intentionally omitted for
now, as the API only needs path-based resolution.
"""

from __future__ import annotations

from functools import cache
from typing import Literal

import fsspec
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from .paths import AnyPathT, PureGSPath, PureR2Path, anypath

FilesystemT = GCSFileSystem | S3FileSystem | LocalFileSystem


@cache
def _filesystem(protocol: Literal["gcs", "gs", "r2", "s3", "local"] = "local") -> FilesystemT:
    """Return a cached filesystem instance for the given protocol.

    Parameters
    ----------
    protocol:
        Storage backend identifier. Supported values are:
        - ``"gcs"`` or ``"gs"`` for Google Cloud Storage
        - ``"r2"`` or ``"s3"`` for Cloudflare R2 or generic S3-compatible storage
        - ``"local"`` for the local filesystem

    Returns
    -------
    FilesystemT
        A concrete fsspec filesystem instance.

    Raises
    ------
    ValueError
        If an unsupported ``protocol`` is provided.
    """
    if protocol in ["gcs", "gs"]:
        return GCSFileSystem()
    if protocol in ["r2", "s3"]:
        return S3FileSystem(anon=False)
    if protocol == "local":
        return fsspec.filesystem("local")

    msg = f"Unknown backend: {protocol}. Supported backends are: gcs, r2, s3, local."
    raise ValueError(msg)


def filesystem_from_path(path: str | AnyPathT) -> FilesystemT:
    """Return an fsspec filesystem appropriate for the given path.

    Parameters
    ----------
    path:
        Local path or cloud URI.

    Returns
    -------
    FilesystemT
        A concrete fsspec filesystem instance that can be used to open the path.
    """
    resolved = anypath(path)

    if isinstance(resolved, PureGSPath):
        return _filesystem("gcs")
    if isinstance(resolved, PureR2Path):
        return _filesystem("r2")

    return _filesystem("local")
