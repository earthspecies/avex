from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from avex.io.paths import PureGSPath
from avex.utils.utils import _get_local_path_for_cloud_file

pytestmark = pytest.mark.skipif(
    os.getuid() == 0,
    reason="root bypasses permission checks",
)


class FakeFS:
    """Minimal fsspec-like FS for cache validation tests."""

    def __init__(self, *, token: str) -> None:
        self._token = token
        self.get_calls: list[tuple[str, str]] = []

    def info(self, _path: str) -> dict[str, Any]:
        return {"etag": self._token, "size": 123}

    def get(self, src: str, dst: str) -> None:
        self.get_calls.append((src, dst))
        Path(dst).write_bytes(b"dummy")

    def set_token(self, token: str) -> None:
        self._token = token


def test_cache_mode_none_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESP_CACHE_HOME", str(tmp_path))
    fs = FakeFS(token="t1")
    path = PureGSPath("gs://bucket/file.pt")

    out = _get_local_path_for_cloud_file(path, fs, "none")

    assert out is None
    assert fs.get_calls == []


def test_cache_use_downloads_then_reuses_when_token_same(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESP_CACHE_HOME", str(tmp_path))
    fs = FakeFS(token="t1")
    path = PureGSPath("gs://bucket/file.pt")

    p1 = _get_local_path_for_cloud_file(path, fs, "use")
    assert p1 is not None and p1.exists()
    assert len(fs.get_calls) == 1

    p2 = _get_local_path_for_cloud_file(path, fs, "use")
    assert p2 == p1
    assert len(fs.get_calls) == 1  # no re-download


def test_cache_use_redownloads_when_remote_token_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESP_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("ESP_CACHE_VALIDATE_TTL_SECONDS", "0")
    fs = FakeFS(token="t1")
    path = PureGSPath("gs://bucket/file.pt")

    p1 = _get_local_path_for_cloud_file(path, fs, "use")
    assert p1 is not None and p1.exists()
    assert len(fs.get_calls) == 1

    fs.set_token("t2")
    p2 = _get_local_path_for_cloud_file(path, fs, "use")
    assert p2 == p1
    assert len(fs.get_calls) == 2  # refreshed due to token mismatch


def test_cache_force_always_redownloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESP_CACHE_HOME", str(tmp_path))
    fs = FakeFS(token="t1")
    path = PureGSPath("gs://bucket/file.pt")

    p1 = _get_local_path_for_cloud_file(path, fs, "force")
    assert p1 is not None and p1.exists()
    assert len(fs.get_calls) == 1

    p2 = _get_local_path_for_cloud_file(path, fs, "force")
    assert p2 == p1
    assert len(fs.get_calls) == 2


def test_failed_download_does_not_leave_corrupt_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESP_CACHE_HOME", str(tmp_path))

    class FailingFS(FakeFS):
        def get(self, src: str, dst: str) -> None:  # type: ignore[override]
            self.get_calls.append((src, dst))
            # Simulate a partial write then failure.
            Path(dst).write_bytes(b"partial")
            raise RuntimeError("network error")

    fs = FailingFS(token="t1")
    path = PureGSPath("gs://bucket/file.pt")

    out = _get_local_path_for_cloud_file(path, fs, "use")
    assert out is None

    # Final cache file should not exist (atomic rename prevents corrupt cache).
    # (Directory name is hashed; just ensure no completed cache artifact exists.)
    assert not any(p.is_file() and p.suffix != ".tmp" for p in tmp_path.rglob("*"))


def test_bucket_is_hashed_in_cache_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESP_CACHE_HOME", str(tmp_path))
    fs = FakeFS(token="t1")
    # Crafted "bucket" that could be problematic if used directly.
    path = PureGSPath("gs://../file.pt")

    out = _get_local_path_for_cloud_file(path, fs, "use")

    assert out is not None
    # Bucket is not used as a directory segment at all; cache path stays under cache root.
    assert out.resolve().is_relative_to(tmp_path.resolve())


def test_cache_unwritable_falls_back_to_direct_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Make cache root non-writable.
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_root.chmod(0o500)  # read/execute only
    monkeypatch.setenv("ESP_CACHE_HOME", str(cache_root))

    fs = FakeFS(token="t1")
    path = PureGSPath("gs://bucket/file.pt")

    out = _get_local_path_for_cloud_file(path, fs, "use")
    assert out is None
