# SPDX-License-Identifier: Apache-2.0
"""Windows-safe file publication for artefacts a dashboard reads while training writes.

CPython opens files without ``FILE_SHARE_DELETE`` on Windows, so ``os.replace``
raises ``PermissionError`` (WinError 5) or a sharing violation (WinError 32) for
as long as *any* process holds the destination open.  The manager dashboard
polls ``metrics.json``, ``manifest.json`` and the league snapshots while the
trainer rewrites them several times per second, so the unguarded
write-temp-then-replace idiom is a live crash vector: a single dashboard refresh
landing on the wrong millisecond aborted a 615k-step run.

Two rules follow, and both are encoded here:

* every publisher retries for a short deadline, because readers hold a file for
  milliseconds and the contention is transient by construction;
* bookkeeping publishers degrade to a warning instead of raising, because model
  checkpoints are the durable artefact of a multi-day run and telemetry must
  never outrank them.

The module deliberately depends on the standard library only, so both the
training package and the dependency-light manager can import it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

__all__ = [
    "DEFAULT_RETRY_SECONDS",
    "atomic_write_json",
    "is_transient_lock_error",
    "replace_with_retry",
    "unlink_with_retry",
]

DEFAULT_RETRY_SECONDS = 10.0
_INITIAL_BACKOFF_SECONDS = 0.01
_MAX_BACKOFF_SECONDS = 0.25

# ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION.
_TRANSIENT_WINERRORS = frozenset({5, 32, 33})


def is_transient_lock_error(exc: BaseException) -> bool:
    """Report whether *exc* is a concurrent reader rather than a real fault.

    A missing source or a full disk must surface immediately; only the
    "someone else has this open" family is worth waiting out.
    """
    winerror = getattr(exc, "winerror", None)
    if winerror is not None:
        return winerror in _TRANSIENT_WINERRORS
    # POSIX never reports this contention, so PermissionError there is genuine
    # and the retry loop simply gives up after the first attempt.
    return os.name == "nt" and isinstance(exc, PermissionError)


def _retry_while_locked(
    operation: Callable[[], None],
    *,
    description: str,
    timeout: float,
    tolerate_failure: bool,
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    backoff = _INITIAL_BACKOFF_SECONDS
    while True:
        try:
            operation()
            return True
        except OSError as exc:
            if not is_transient_lock_error(exc):
                raise
            if time.monotonic() >= deadline:
                if tolerate_failure:
                    print(
                        f"[nextocr] skipped {description}: still locked by another "
                        f"process after {timeout:.1f}s ({exc})",
                        file=sys.stderr,
                        flush=True,
                    )
                    return False
                raise
            time.sleep(backoff)
            backoff = min(_MAX_BACKOFF_SECONDS, backoff * 2)


def replace_with_retry(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    timeout: float = DEFAULT_RETRY_SECONDS,
    tolerate_failure: bool = False,
) -> bool:
    """``os.replace`` that waits out concurrent readers of *destination*.

    Returns ``True`` when the replacement landed.  Only returns ``False`` when
    *tolerate_failure* is set and the destination stayed locked for the whole
    deadline.
    """
    return _retry_while_locked(
        lambda: os.replace(source, destination),
        description=f"publishing {os.fspath(destination)}",
        timeout=timeout,
        tolerate_failure=tolerate_failure,
    )


def unlink_with_retry(
    path: str | os.PathLike[str],
    *,
    timeout: float = DEFAULT_RETRY_SECONDS,
    tolerate_failure: bool = True,
) -> bool:
    """Delete *path*, waiting out a reader that still has it open.

    Defaults to tolerating failure: an orphaned file costs disk space, while
    raising would abort training.
    """
    target = Path(path)

    def _unlink() -> None:
        target.unlink(missing_ok=True)

    return _retry_while_locked(
        _unlink,
        description=f"deleting {target}",
        timeout=timeout,
        tolerate_failure=tolerate_failure,
    )


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    timeout: float = DEFAULT_RETRY_SECONDS,
    tolerate_failure: bool = False,
    fsync: bool = False,
    indent: int | None = 2,
    sort_keys: bool = True,
    separators: tuple[str, str] | None = None,
) -> bool:
    """Serialise *value* to a sibling temp file, then publish it over *path*.

    The temp file carries the writer's pid so two processes publishing the same
    artefact cannot clobber each other's partial writes.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=False,
                separators=separators,
            )
            handle.write("\n")
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        published = replace_with_retry(
            temporary,
            destination,
            timeout=timeout,
            tolerate_failure=tolerate_failure,
        )
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return published
