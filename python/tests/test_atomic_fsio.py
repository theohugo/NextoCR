# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the file publication that a live dashboard contends with.

The motivating incident: the manager dashboard polled ``metrics.json`` while the
trainer republished it, ``os.replace`` raised ``PermissionError`` (WinError 5),
and a 615k-step training run died inside ``model.learn``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nextocr_fsio import (
    atomic_write_json,
    is_transient_lock_error,
    replace_with_retry,
    unlink_with_retry,
)


WINDOWS_ONLY = pytest.mark.skipif(
    os.name != "nt", reason="POSIX allows replacing a file that readers hold open"
)


def _permission_error(winerror: int) -> PermissionError:
    error = PermissionError(13, "Access is denied")
    error.winerror = winerror
    return error


def test_atomic_write_json_round_trips(tmp_path: Path) -> None:
    destination = tmp_path / "metrics.json"

    assert atomic_write_json(destination, {"steps": 615682}) is True

    assert json.loads(destination.read_text(encoding="utf-8")) == {"steps": 615682}


def test_atomic_write_json_leaves_no_temporary_files(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"

    atomic_write_json(destination, {"status": "running"})

    assert [path.name for path in tmp_path.iterdir()] == ["manifest.json"]


def test_sharing_violation_is_transient_but_missing_file_is_not() -> None:
    assert is_transient_lock_error(_permission_error(5)) is True
    assert is_transient_lock_error(_permission_error(32)) is True
    # A genuine fault must surface immediately instead of burning the deadline.
    assert is_transient_lock_error(FileNotFoundError(2, "No such file")) is False
    assert is_transient_lock_error(OSError(28, "No space left on device")) is False


def test_retry_gives_up_and_reports_failure_when_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    def always_locked(*_args: object, **_kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        raise _permission_error(5)

    monkeypatch.setattr(os, "replace", always_locked)
    source = tmp_path / "source"
    source.write_text("payload", encoding="utf-8")

    published = replace_with_retry(
        source, tmp_path / "destination", timeout=0.05, tolerate_failure=True
    )

    assert published is False
    assert attempts > 1, "a transient lock must be retried, not failed on first sight"


def test_retry_raises_when_failure_is_not_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        os, "replace", lambda *_a, **_k: (_ for _ in ()).throw(_permission_error(32))
    )
    source = tmp_path / "source"
    source.write_text("payload", encoding="utf-8")

    with pytest.raises(PermissionError):
        replace_with_retry(source, tmp_path / "destination", timeout=0.05)


def test_replace_succeeds_once_the_reader_releases_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = os.replace
    remaining_locks = 3

    def locked_until_reader_closes(source: object, destination: object) -> None:
        nonlocal remaining_locks
        if remaining_locks > 0:
            remaining_locks -= 1
            raise _permission_error(5)
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", locked_until_reader_closes)
    destination = tmp_path / "metrics.json"

    assert atomic_write_json(destination, {"steps": 700000}) is True

    assert remaining_locks == 0
    assert json.loads(destination.read_text(encoding="utf-8")) == {"steps": 700000}


@WINDOWS_ONLY
def test_open_reader_does_not_abort_publication(tmp_path: Path) -> None:
    """The exact incident: publish while a dashboard-style reader holds the file."""
    destination = tmp_path / "metrics.json"
    atomic_write_json(destination, {"steps": 1})

    with destination.open("r", encoding="utf-8") as reader:
        reader.read()
        # Windows refuses the replace for as long as this handle lives, so the
        # short deadline must expire and the trainer must survive it.
        assert (
            atomic_write_json(
                destination, {"steps": 2}, timeout=0.2, tolerate_failure=True
            )
            is False
        )

    # Once the reader closes, the next flush lands normally.
    assert atomic_write_json(destination, {"steps": 3}) is True
    assert json.loads(destination.read_text(encoding="utf-8")) == {"steps": 3}


@WINDOWS_ONLY
def test_league_snapshot_deletion_survives_an_open_reader(tmp_path: Path) -> None:
    snapshot = tmp_path / "policy_000012.zip"
    snapshot.write_bytes(b"model")

    with snapshot.open("rb") as reader:
        reader.read()
        assert unlink_with_retry(snapshot, timeout=0.2) is False

    assert unlink_with_retry(snapshot) is True
    assert not snapshot.exists()
