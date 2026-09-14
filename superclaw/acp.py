from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from superclaw import __version__
from superclaw.app import Callbacks, Runtime, run_once
from superclaw.policy import CYCLE_MODES, Mode

PROTOCOL_VERSION = 1
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR, BUSY = -32700, -32600, -32601, -32602, -32603, -32000
KINDS = {"read_file": "read", "list_directory": "read", "glob": "search", "grep": "search", "write_file": "edit", "edit_file": "edit",
         "bash": "execute", "python": "execute", "update_plan": "think", "delegate": "think"}
OPTIONS = (("allow", "allow_once", "Allow"), ("allow_session", "allow_always", "Allow for this session"),
           ("allow_prefix", "allow_always", "Remember this command prefix"), ("deny", "reject_once", "Deny"))
STOP = {"cancelled": "cancelled", "max_turns": "max_turn_requests", "max_tokens": "max_tokens"}


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _data_url(block: dict[str, Any]) -> str:
    return f"data:{block.get('mimeType') or 'image/png'};base64,{block.get('data', '')}"


def prompt_of(blocks: list[dict[str, Any]]) -> tuple[str, list[str]]:
    texts, images = [], []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            texts.append(str(block.get("text", "")))
        elif kind == "image":
            images.append(_data_url(block))
        elif kind == "resource":
            resource = block.get("resource") or {}
            texts.append(f'<attachment path="{resource.get("uri", "")}">\n{resource.get("text", "")}\n</attachment>')
        elif kind == "resource_link":
            texts.append(f"Referenced file: {block.get('uri', '')}")
    return "\n\n".join(t for t in texts if t), images


def tool_title(name: str, args: dict[str, Any]) -> str:
    target = args.get("path") or args.get("pattern") or args.get("command") or args.get("task") or args.get("query") or ""
    return f"{name} {target}".strip()


def tool_content(event: dict[str, Any]) -> list[dict[str, Any]]:
    display = event.get("display") or {}
    if display.get("kind") == "diff" and display.get("preview"):
        return [{"type": "content", "content": {"type": "text", "text": display["preview"]}}]
    return [{"type": "content", "content": {"type": "text", "text": event.get("output", "")}}]


@dataclass
class Turn:
    session_id: str
    cancel: threading.Event = field(default_factory=threading.Event)
    last_call: str = ""
    streamed: bool = False


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
        for line in self._reader:
            line = line.strip()
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
                worker = threading.Thread(target=self._handle, args=(message,), daemon=True, name="acp-request")
                self._inflight = [t for t in self._inflight if t.is_alive()] + [worker]
                worker.start()
        for future in self._pending.values():
            future.set_result({"error": {"code": INTERNAL_ERROR, "message": "client closed the connection"}})
        for worker in self._inflight:
            worker.join()


class Agent:
    def __init__(self, rt: Runtime, conn: Conn) -> None:
        self.rt = rt
        self.conn = conn
        self.turns: dict[str, Turn] = {}
        self.busy = threading.Lock()
        conn.handlers.update({
            "initialize": self.initialize, "session/new": self.new_session, "session/load": self.load_session,
            "session/list": self.list_sessions, "session/prompt": self.prompt, "session/set_mode": self.set_mode,
        })
        conn.notifiers["session/cancel"] = self.cancel

    def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {"loadSession": True, "promptCapabilities": {"image": True, "audio": False, "embeddedContext": True},
                                  "sessionCapabilities": {"list": {}}},
            "agentInfo": {"name": "superclaw", "version": __version__},
            "authMethods": [],
        }

    def _modes(self) -> dict[str, Any]:
        return {"currentModeId": self.rt.mode.value,
                "availableModes": [{"id": m.value, "name": m.value} for m in (*CYCLE_MODES, Mode.UNSAFE)]}

    def _workspace(self, cwd: Any) -> Path:
        path = Path(str(cwd or ""))
        if not path.is_absolute() or not path.is_dir():
            raise RpcError(INVALID_PARAMS, f"cwd must be an absolute directory, got {cwd!r}")
        if path.resolve() != self.rt.workspace.resolve():
            raise RpcError(INVALID_PARAMS, f"this superclaw serves {self.rt.workspace}; start one there or pass -C")
        return path

    def new_session(self, params: dict[str, Any]) -> dict[str, Any]:
        self._workspace(params.get("cwd"))
        sid = self.rt.store.create(cwd=str(self.rt.workspace), model=self.rt.model)
        self.turns[sid] = Turn(sid)
        return {"sessionId": sid, "modes": self._modes()}

    def load_session(self, params: dict[str, Any]) -> dict[str, Any]:
        self._workspace(params.get("cwd"))
        sid = str(params.get("sessionId") or "")
        if self.rt.store.get(sid) is None:
            raise RpcError(INVALID_PARAMS, f"unknown session {sid!r}")
        self.turns[sid] = Turn(sid)
        for message in self.rt.store.replay(sid):
            if message.role in ("user", "assistant") and message.content:
                self.update(sid, {"sessionUpdate": f"{'user' if message.role == 'user' else 'agent'}_message_chunk",
                                  "content": {"type": "text", "text": message.content}})
        return {"modes": self._modes()}

    def list_sessions(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"sessions": [{"sessionId": s["id"], "cwd": s["cwd"], "title": s["title"]} for s in self.rt.store.recent()]}

    def set_mode(self, params: dict[str, Any]) -> dict[str, Any]:
        wanted = str(params.get("modeId") or "")
        if wanted not in Mode._value2member_map_:
            raise RpcError(INVALID_PARAMS, f"unknown mode {wanted!r}")
        self.rt.mode = Mode(wanted)
        self.update(str(params.get("sessionId") or ""), {"sessionUpdate": "current_mode_update", "currentModeId": wanted})
        return {}

    def cancel(self, params: dict[str, Any]) -> None:
        turn = self.turns.get(str(params.get("sessionId") or ""))
        if turn is not None:
            turn.cancel.set()

    def update(self, sid: str, update: dict[str, Any]) -> None:
        self.conn.notify("session/update", {"sessionId": sid, "update": update})

    def on_event(self, turn: Turn, event: dict[str, Any]) -> None:
        kind = event["type"]
        if event.get("child"):
            return
        if kind == "text_delta":
            turn.streamed = True
            self.update(turn.session_id, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": event["text"]}})
        elif kind == "text":
            if not turn.streamed:
                self.update(turn.session_id, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": event["text"]}})
            turn.streamed = False
        elif kind == "tool_call":
            turn.last_call = event["id"]
            self.update(turn.session_id, {"sessionUpdate": "tool_call", "toolCallId": event["id"], "title": tool_title(event["name"], event["args"]),
                                          "kind": KINDS.get(event["name"], "other"), "status": "in_progress", "rawInput": event["args"]})
        elif kind == "tool_result":
            self.update(turn.session_id, {"sessionUpdate": "tool_call_update", "toolCallId": event["id"],
                                          "status": "completed" if event["ok"] else "failed", "content": tool_content(event)})

    def on_permission(self, turn: Turn, request: dict[str, Any]) -> str:
        offered = [(action, kind, name) for action, kind, name in OPTIONS if action != "allow_prefix" or request.get("prefix")]
        try:
            reply = self.conn.request("session/request_permission", {
                "sessionId": turn.session_id,
                "toolCall": {"toolCallId": turn.last_call, "title": tool_title(request["tool"], request["args"]),
                             "kind": KINDS.get(request["tool"], "other"), "status": "pending", "rawInput": request["args"]},
                "options": [{"optionId": action, "kind": kind, "name": name} for action, kind, name in offered],
            })
        except RpcError:
            return "deny"
        outcome = reply.get("outcome") or {}
        if outcome.get("outcome") == "cancelled":
            turn.cancel.set()
            return "deny"
        chosen = str(outcome.get("optionId") or "")
        return chosen if any(chosen == action for action, _, _ in offered) else "deny"

    def prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        sid = str(params.get("sessionId") or "")
        turn = self.turns.get(sid)
        if turn is None:
            raise RpcError(INVALID_PARAMS, f"unknown session {sid!r}; call session/new or session/load first")
        text, images = prompt_of(list(params.get("prompt") or []))
        if not text.strip() and not images:
            raise RpcError(INVALID_PARAMS, "prompt has no text")
        if not self.busy.acquire(blocking=False):
            raise RpcError(BUSY, "a prompt is already running")
        try:
            turn.cancel.clear()
            turn.streamed = False
            result = run_once(self.rt, text, sid, Callbacks(on_event=lambda e: self.on_event(turn, e), on_permission=lambda r: self.on_permission(turn, r)),
                              cancelled=turn.cancel.is_set, images=images)
        finally:
            self.busy.release()
        return {"stopReason": STOP.get(result.stop_reason, "end_turn")}


def serve(rt: Runtime, reader: IO[str] = sys.stdin, writer: IO[str] = sys.stdout) -> None:
    conn = Conn(reader, writer)
    Agent(rt, conn)
    conn.serve()
