from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
from pathlib import Path
from typing import Any

from supergraph import SuperGraph
from supergraph.core.errors import StoreInUse, SuperGraphError
from supergraph.core.types import Result, encode_json

from superclaw.settings import LIMITS, SHARE_DIR


class NotServing(StoreInUse):
    def __init__(self, lock_path: Path, sock: Path) -> None:
        SuperGraphError.__init__(self, f"{lock_path} is held by a superclaw that is not serving it at {sock}; "
                                       "it predates store sharing, so restart that session")
        self.lock_path = str(lock_path)


def socket_path(db_path: Path) -> Path:
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", "").strip() or f"/tmp/superclaw-{os.getuid()}")
    key = hashlib.sha256(str(Path(db_path).resolve()).encode()).hexdigest()[: LIMITS.share_key_chars]
    return runtime / SHARE_DIR / f"{key}.sock"


class Server:
    def __init__(self, gs: Any, path: Path) -> None:
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
            try:
                self._listener.close()
            except OSError:
                pass
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
            try:
                sock.close()
            except OSError:
                pass


class SharedGraph:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._owned: Any = None
        self._server: Server | None = None
        self._remote: RemoteGraph | None = None
        self._switch = threading.Lock()

    @property
    def role(self) -> str:
        return "owner" if self._owned is not None else "attached"

    def _own(self) -> None:
        self._owned = SuperGraph(path=str(self.db_path), queued=True)
        self._server = Server(self._owned, socket_path(self.db_path))
        self._server.start()

    def open(self) -> SharedGraph:
        try:
            self._own()
        except StoreInUse:
            path = socket_path(self.db_path)
            if not path.exists():
                raise NotServing(self.db_path / ".supergraph.lock", path) from None
            self._remote = RemoteGraph(path)
        return self

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
                self._remote = RemoteGraph(socket_path(self.db_path))
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


def open_shared(db_path: Path) -> SharedGraph:
    return SharedGraph(db_path).open()
