"""Single-owner lock file for persistent SuperGraph paths.

Two SuperGraph instances opening the same path concurrently would race on
WAL replay + checkpoint + sqlite WAL-mode transactions. SQLite handles
single-writer concurrency at the row level but the compact/snapshot/
optimizer paths are not cross-process safe.

This module takes an advisory ``fcntl.flock`` (Unix) or ``msvcrt.locking``
(Windows) on ``<path>/.supergraph.lock``. Fresh acquisition wins; a second
caller raises ``StoreInUse``. Lock releases on ``gs.close()`` and on
process exit (OS cleans up the fd).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


LOCK_FILENAME = ".supergraph.lock"


class _LockHandle:
    """Opaque handle. Caller keeps ref; close() releases the lock."""

    def __init__(self, fd: int, path: str):
        self.fd = fd
        self.path = path
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    os.lseek(self.fd, 0, 0)
                    msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            try:
                os.close(self.fd)
            except OSError:
                pass


def acquire_path_lock(db_dir: str | Path) -> Optional[_LockHandle]:
    """Try to take an exclusive advisory lock on db_dir/.supergraph.lock.

    Returns a handle the caller must retain and .release() on close.
    Raises ``StoreInUse`` when another process holds the lock.
    Returns ``None`` on platforms where locking is unavailable.
    """
    from supergraph.core.errors import StoreInUse

    lock_path = Path(db_dir) / LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Open RW creates the file if missing. O_CLOEXEC prevents leaking to
    # child processes (they'd inherit the lock).
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(str(lock_path), flags, 0o644)

    try:
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError as e:
                os.close(fd)
                raise StoreInUse(str(lock_path)) from e
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                os.close(fd)
                raise StoreInUse(str(lock_path)) from e
    except ImportError:
        # Locking primitive not importable (rare) - degrade to unlocked.
        return None

    # Record PID for ops-visibility; non-authoritative (not used for unlock).
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
    except OSError:
        pass

    return _LockHandle(fd, str(lock_path))
