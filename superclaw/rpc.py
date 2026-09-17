from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import IO, Any

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR, BUSY = -32700, -32600, -32601, -32602, -32603, -32000


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class Conn:
    def __init__(self, reader: IO[str], writer: IO[str]) -> None:
        self._reader = reader
        self._writer = writer
        self._lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, Future[dict[str, Any]]] = {}
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.notifiers: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._inflight: list[threading.Thread] = []

    def _write(self, message: dict[str, Any]) -> None:
        with self._lock:
            self._writer.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
            self._writer.flush()

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"method": method, "params": params})

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            message_id = self._next_id
            self._pending[message_id] = Future()
        self._write({"id": message_id, "method": method, "params": params})
        reply = self._pending[message_id].result()
        if reply.get("error"):
            raise RpcError(int(reply["error"].get("code", INTERNAL_ERROR)), str(reply["error"].get("message", "")))
        return reply.get("result") or {}

    def _reply(self, message_id: Any, result: Any = None, error: RpcError | None = None) -> None:
        if error is not None:
            self._write({"id": message_id, "error": {"code": error.code, "message": str(error)}})
        else:
            self._write({"id": message_id, "result": result if result is not None else {}})

    def _handle(self, message: dict[str, Any]) -> None:
        method, message_id, params = message.get("method"), message.get("id"), message.get("params") or {}
        handler = self.handlers.get(str(method))
        if handler is None:
            self._reply(message_id, error=RpcError(METHOD_NOT_FOUND, f"unknown method {method!r}"))
            return
        try:
            self._reply(message_id, handler(params))
        except RpcError as e:
            self._reply(message_id, error=e)
        except Exception as e:
            self._reply(message_id, error=RpcError(INTERNAL_ERROR, f"{type(e).__name__}: {e}"))

    def serve(self) -> None:
        for raw in self._reader:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                self._reply(None, error=RpcError(PARSE_ERROR, "invalid JSON"))
                continue
            if not isinstance(message, dict):
                self._reply(None, error=RpcError(INVALID_REQUEST, "expected an object"))
            elif "method" not in message and message.get("id") is not None:
                future = self._pending.pop(int(message["id"]), None)
                if future is not None:
                    future.set_result(message)
            elif message.get("id") is None:
                notifier = self.notifiers.get(str(message.get("method")))
                if notifier is not None:
                    notifier(message.get("params") or {})
            else:
                worker = threading.Thread(target=self._handle, args=(message,), daemon=True, name="rpc-request")
                self._inflight = [t for t in self._inflight if t.is_alive()] + [worker]
                worker.start()
        for future in self._pending.values():
            future.set_result({"error": {"code": INTERNAL_ERROR, "message": "client closed the connection"}})
        for worker in self._inflight:
            worker.join()
