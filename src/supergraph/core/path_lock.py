
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


LOCK_FILENAME = ".supergraph.lock"


class _LockHandle:

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
    from supergraph.core.errors import StoreInUse

    lock_path = Path(db_dir) / LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
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
        return None

    try:
        os.write(fd, f"{os.getpid()}\n".encode())
    except OSError:
        pass

    return _LockHandle(fd, str(lock_path))
