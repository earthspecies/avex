"""Path utilities for cloud and local URIs.

This module provides a minimal subset of the ``esp_data.io.paths`` interface that is
required by the public API. It defines lightweight "pure" path classes for cloud URIs
and an ``anypath`` helper that converts strings to the appropriate path type.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeAlias


class PureCloudPath:
    """Base class for cloud path manipulation.

    This is a simplified version of the ``esp_data`` implementation. It focuses on the
    behaviours required by the public API:

    - Representing URIs like ``gs://bucket/path/to/file`` or ``s3://bucket/key``
    - Basic string conversion
    - Bucket extraction
    - Simple parent navigation

    It intentionally does not aim to be a full pathlib drop-in replacement.
    """

    cloud_prefix: str = ""
    __slots__ = ("_path",)

    def __init__(self, path: str) -> None:
        """Create a new cloud path.

        Parameters
        ----------
        path:
            Cloud URI starting with the appropriate ``cloud_prefix``.

        Raises
        ------
        ValueError
            If ``cloud_prefix`` is not defined, ``path`` is empty, or ``path`` does
            not start with the expected ``cloud_prefix``.
        """
        if not self.cloud_prefix:
            msg = "cloud_prefix must be defined in subclass"
            raise ValueError(msg)

        if not path:
            msg = "Path is empty"
            raise ValueError(msg)

        if not path.startswith(self.cloud_prefix):
            msg = f"Path must start with '{self.cloud_prefix}': {path}"
            raise ValueError(msg)

        self._path = path

    def __str__(self) -> str:
        """Return the string representation of the path.

        Returns
        -------
        str
            The underlying cloud path string.
        """
        return self._path

    def __repr__(self) -> str:
        """Return a readable representation of the path.

        Returns
        -------
        str
            Debug-friendly representation including the class name and path.
        """
        return f"{self.__class__.__name__}({self._path!r})"

    def __fspath__(self) -> str:
        """Return the file system path representation.

        Returns
        -------
        str
            The path string suitable for use with file system APIs.
        """
        return self._path

    @property
    def name(self) -> str:
        """Return the final component of the path."""
        return os.path.basename(self._path.rstrip("/"))

    @property
    def bucket(self) -> str:
        """Return the bucket component of the path."""
        remainder = self._path[len(self.cloud_prefix) :]
        if "/" in remainder:
            return remainder.split("/")[0]
        return remainder

    @property
    def stem(self) -> str:
        """Return the final component without its suffix."""
        name = self.name
        return os.path.splitext(name)[0]


class PureGSPath(PureCloudPath):
    """Google Cloud Storage path."""

    cloud_prefix = "gs://"
    __slots__ = ()


class PureS3Path(PureCloudPath):
    """Amazon S3 path."""

    cloud_prefix = "s3://"
    __slots__ = ()


class PureR2Path(PureCloudPath):
    """Cloudflare R2 path (S3-compatible)."""

    cloud_prefix = "s3://"
    __slots__ = ()


class PureHFPath(PureCloudPath):
    """Hugging Face Hub path.

    This uses the `huggingface_hub` fsspec-compatible filesystem under the hood.

    Expected format:
        hf://<repo_id>/<path>
    """

    cloud_prefix = "hf://"
    __slots__ = ()


AnyPathT: TypeAlias = Path | PureGSPath | PureR2Path | PureS3Path | PureHFPath


def anypath(path: str | AnyPathT) -> AnyPathT:
    """Create the appropriate path object based on the input.

    This mirrors the subset of ``esp_data.io.anypath`` that is relied upon by the
    public API. It supports:

    - Local paths -> ``pathlib.Path``
    - ``gs://`` URIs -> ``PureGSPath``
    - ``r2://`` and ``s3://`` URIs -> ``PureR2Path``

    Parameters
    ----------
    path:
        Local path or cloud URI.

    Returns
    -------
    AnyPathT
        A concrete path instance suitable for IO operations.
    """
    as_str = str(path)

    if as_str.startswith("gs://"):
        return PureGSPath(as_str)
    if as_str.startswith("r2://"):
        return PureR2Path("s3://" + as_str.removeprefix("r2://"))
    if as_str.startswith("s3://"):
        return PureR2Path(as_str)
    if as_str.startswith("hf://"):
        return PureHFPath(as_str)

    return Path(as_str)
