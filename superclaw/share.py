from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from superclaw.dsl import Store
from superclaw.settings import LIMITS, PROC_DIR, SHARE_DIR, STOPPED_STATES
from supergraph import SuperGraph
from supergraph.core.errors import StoreInUse, SuperGraphError
from supergraph.core.path_lock import LOCK_FILENAME
from supergraph.core.types import Result, encode_json


def holder(lock_path: Path) -> tuple[int, str]:
    try:
        pid = int(lock_path.read_text().split()[0])
        stat = (Path(PROC_DIR) / str(pid) / "stat").read_text()
    except (OSError, ValueError, IndexError):
        return 0, ""
    return pid, stat.rpartition(")")[2].split()[0]


class NotServing(StoreInUse):
    def __init__(self, lock_path: Path, sock: Path) -> None:
        pid, state = holder(lock_path)
        advice = (f"superclaw {pid} holds it but is suspended; resume that terminal with fg, or end it with kill {pid}"
                  if state.startswith(STOPPED_STATES) else "it predates store sharing, so restart that session")
        SuperGraphError.__init__(self, f"{lock_path} is not being served at {sock}; {advice}")
        self.lock_path = str(lock_path)


def socket_path(db_path: Path) -> Path:
    fallback = Path(tempfile.gettempdir()) / f"superclaw-{os.getuid()}"
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", "").strip() or fallback)
    key = hashlib.sha256(str(Path(db_path).resolve()).encode()).hexdigest()[: LIMITS.share_key_chars]
    return runtime / SHARE_DIR / f"{key}.sock"


class Server:
    def __init__(self, gs: Store, path: Path) -> None:
        self.gs = gs
        self.path = path
        self._listener: socket.socket | None = None
        self._closed = threading.Event()
        self._connections: set[socket.socket] = set()
        self._guard = threading.Lock()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.path))
        os.chmod(self.path, 0o600)
        listener.listen(LIMITS.share_backlog)
        self._listener = listener
        threading.Thread(target=self._accept, daemon=True, name="superclaw-share").start()

    def _accept(self) -> None:
        assert self._listener is not None
        while not self._closed.is_set():
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            with self._guard:
                self._connections.add(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True, name="superclaw-share-conn").start()

    def _serve(self, conn: socket.socket) -> None:
        with conn, conn.makefile("rb") as reader:
            for line in reader:
                if len(line) > LIMITS.share_message_bytes or self._closed.is_set():
                    break
                try:
                    request = json.loads(line)
                    result = self.gs.execute(str(request["query"]), namespace=request.get("namespace"))
                    payload = {"kind": result.kind, "data": result.data, "count": result.count}
                except Exception as e:
                    payload = {"error": {"type": type(e).__name__, "message": str(e)}}
                try:
                    conn.sendall(encode_json(payload) + b"\n")
                except OSError:
                    return

    def close(self) -> None:
        self._closed.set()
        if self._listener is not None:
            with contextlib.suppress(OSError):
                self._listener.close()
        with self._guard:
            connections, self._connections = list(self._connections), set()
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
            except OSError:
                pass
        self.path.unlink(missing_ok=True)


class RemoteGraph:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(LIMITS.share_timeout_s)
        sock.connect(str(self.path))
        return sock

    def execute(self, query: str, *, namespace: str | None = None) -> Result:
        with self._lock:
            if self._sock is None:
                self._sock = self._connect()
            try:
                self._sock.sendall(json.dumps({"query": query, "namespace": namespace}).encode() + b"\n")
                line = self._sock.makefile("rb").readline()
            except OSError:
                self.close()
                raise
            if not line:
                self.close()
                raise ConnectionError("the superclaw serving the brain closed the connection")
        reply = json.loads(line)
        if "error" in reply:
            raise SuperGraphError(f"{reply['error']['type']}: {reply['error']['message']}")
        return Result(kind=reply["kind"], data=reply["data"], count=int(reply["count"]))

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()


class SharedGraph:
    def __init__(self, db_path: Path, reader: Callable[[str], str] | None = None) -> None:
        self.db_path = Path(db_path)
        self._reader = reader
        self._owned: Any = None
        self._server: Server | None = None
        self._remote: RemoteGraph | None = None
        self._switch = threading.Lock()

    @property
    def role(self) -> str:
        return "owner" if self._owned is not None else "attached"

    def _own(self) -> None:
        self._owned = SuperGraph(path=str(self.db_path), queued=True, reader=self._reader)
        self._server = Server(self._owned, socket_path(self.db_path))
        self._server.start()

    def open(self) -> SharedGraph:
        try:
            self._own()
        except StoreInUse:
            self._remote = self._attach()
        return self

    def _attach(self) -> RemoteGraph:
        path = socket_path(self.db_path)
        lock = self.db_path / LOCK_FILENAME
        deadline = time.monotonic() + LIMITS.share_attach_timeout_s
        while True:
            if path.exists():
                remote = RemoteGraph(path)
                try:
                    remote.execute("SYS HEALTH")
                    return remote
                except (OSError, ConnectionError):
                    remote.close()
            if holder(lock)[1].startswith(STOPPED_STATES) or time.monotonic() >= deadline:
                raise NotServing(lock, path) from None
            time.sleep(LIMITS.share_attach_poll_s)

    def _takeover(self) -> None:
        with self._switch:
            if self._owned is not None:
                return
            remote, self._remote = self._remote, None
            if remote is not None:
                remote.close()
            try:
                self._own()
            except StoreInUse as e:
                self._remote = self._attach()
                raise SuperGraphError(f"another superclaw holds the brain but is not answering on {socket_path(self.db_path)}") from e

    def execute(self, query: str, *, namespace: str | None = None) -> Result:
        if self._owned is not None:
            return self._owned.execute(query, namespace=namespace)
        assert self._remote is not None
        try:
            return self._remote.execute(query, namespace=namespace)
        except (ConnectionError, OSError):
            self._takeover()
            return self.execute(query, namespace=namespace)

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        if self._remote is not None:
            self._remote.close()
            self._remote = None
        if self._owned is not None:
            self._owned.close()
            self._owned = None


def open_shared(db_path: Path, reader: Callable[[str], str] | None = None) -> SharedGraph:
    return SharedGraph(db_path, reader).open()
