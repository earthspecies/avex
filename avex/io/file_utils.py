"""High-level file utilities built on top of the internal filesystem."""

from __future__ import annotations

from typing import Any

from .filesystem import filesystem_from_path
from .paths import AnyPathT


def exists(path: str | AnyPathT) -> bool:
    """Return whether a file or directory exists.

    Parameters
    ----------
    path:
        File or directory path to check.

    Returns
    -------
    bool
        ``True`` if the path exists, ``False`` otherwise.
    """
    fs = filesystem_from_path(path)
    return fs.exists(str(path))


def rm(
    path: str | AnyPathT,
    recursive: bool = False,
    maxdepth: int | None = None,
    **kwargs: dict[str, Any],
) -> None:
    """Delete files or directories.

    Parameters
    ----------
    path:
        File or directory path to delete.
    recursive:
        If ``True`` and the path is a directory, delete contents recursively.
    maxdepth:
        Maximum recursion depth for directory deletion. If ``None``, there is
        no explicit depth limit.
    **kwargs:
        Additional keyword arguments forwarded to the underlying filesystem
        ``rm`` implementation.
    """
    fs = filesystem_from_path(path)
    fs.rm(str(path), recursive=recursive, maxdepth=maxdepth, **kwargs)
