from __future__ import annotations

import json
import os
import queue
import re
import signal

from superclaw.hooks import substitute
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from superclaw import __version__
from superclaw.settings import CLAUDE_MCP_FILE, LIMITS
from superclaw.tools import Permission, Registry, Result, Safety, SideEffect, Tool, ToolContext

Source = tuple[str, dict[str, Any], Path, set[str]]

PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "superclaw"
SERVER_KEYS = ("mcpServers", "servers")
TOOL_PREFIX = "mcp_"
EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": True}
_UNSAFE = re.compile(r"[^a-z0-9_]+")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class MCPError(RuntimeError):
    pass


@dataclass(frozen=True)
class Server:
    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def transport(self) -> str:
        return "http" if self.url else "stdio"


@dataclass
class Config:
    servers: list[Server] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _part(value: str) -> str:
    return _UNSAFE.sub("_", value.strip().lower().replace("-", "_")).strip("_")


def tool_name(server: str, remote: str) -> str:
    return f"{TOOL_PREFIX}{_part(server)}_{_part(remote) or 'tool'}"


def _servers_of(data: Any) -> dict[str, Any]:
    for key in SERVER_KEYS:
        block = data.get(key) if isinstance(data, dict) else None
        if isinstance(block, dict):
            return block
    return {}


def _read_json(path: Path, problems: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        problems.append(f"{path}: {type(e).__name__}: {e}")
        return {}
    return data if isinstance(data, dict) else {}


def claude_sources(state: Path, workspace: Path, trusted: bool) -> list[Source]:
    problems: list[str] = []
    data = _read_json(state, problems)
    project = (data.get("projects") or {}).get(str(workspace)) or {}
    disabled = set(data.get("disabledMcpServers") or []) | set(project.get("disabledMcpServers") or [])
    sources: list[Source] = [(str(state), _servers_of(data), state.parent, disabled), (f"{state}#{workspace}", _servers_of(project), workspace, disabled)]
    if trusted:
        sources.append((str(workspace / CLAUDE_MCP_FILE), _servers_of(_read_json(workspace / CLAUDE_MCP_FILE, problems)), workspace,
                        set(project.get("disabledMcpjsonServers") or [])))
    return sources


def _pairs(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise MCPError(f"{what} must look like KEY=VALUE, got {item!r}")
        out[key.strip()] = value
    return out


def add_server(path: Path, name: str, command: list[str], url: str = "", env: list[str] | None = None, headers: list[str] | None = None) -> Server:
    name = name.strip()
    if not _NAME.match(name):
        raise MCPError(f"invalid server name {name!r}; use letters, digits, - and _")
    raw: dict[str, Any] = {}
    if command:
        raw.update(command=command[0], args=command[1:])
    if url:
        raw["url"] = url
    if env:
        raw["env"] = _pairs(env, "--env")
    if headers:
        raw["headers"] = _pairs(headers, "--header")
    if problem := _validate(name, raw):
        raise MCPError(problem)
    problems: list[str] = []
    data = _read_json(path, problems)
    if problems:
        raise MCPError(problems[0])
    key = next((k for k in SERVER_KEYS if isinstance(data.get(k), dict)), SERVER_KEYS[0])
    servers = data.setdefault(key, {})
    if name in servers:
        raise MCPError(f"server {name!r} already exists in {path}; remove it first")
    servers[name] = {k: v for k, v in raw.items() if v}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return load_config([path]).servers[[s.name for s in load_config([path]).servers].index(name)]


def remove_server(path: Path, name: str) -> None:
    problems: list[str] = []
    data = _read_json(path, problems)
    if problems:
        raise MCPError(problems[0])
    key = next((k for k in SERVER_KEYS if isinstance(data.get(k), dict) and name in data[k]), "")
    if not key:
        raise MCPError(f"no server {name!r} in {path}")
    del data[key][name]
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_config(entries: list[Path | Source]) -> Config:
    config = Config()
    seen: set[str] = set()
    for entry in entries:
        if isinstance(entry, Path):
            data = _read_json(entry, config.problems)
            servers, root, disabled = _servers_of(data), entry.parent, set()
        else:
            _, servers, root, disabled = entry
        for name, raw in sorted(servers.items()):
            if not isinstance(raw, dict) or raw.get("disabled") or name in disabled:
                continue
            problem = _validate(name, raw)
            if problem:
                config.problems.append(problem)
                continue
            if name in seen:
                config.problems.append(f"{name}: already defined in an earlier config file; the first one wins")
                continue
            seen.add(name)
            config.servers.append(Server(
                name=name, command=substitute(str(raw.get("command") or "").strip(), root),
                args=[substitute(str(a), root) for a in raw.get("args") or []],
                env={str(k): substitute(str(v), root) for k, v in (raw.get("env") or {}).items()},
                url=str(raw.get("url") or "").strip(),
                headers={str(k): str(v) for k, v in (raw.get("headers") or {}).items()},
            ))
    return config


def _validate(name: str, raw: dict[str, Any]) -> str:
    kind = str(raw.get("type") or "").strip().lower() or ("http" if raw.get("url") else "stdio")
    if kind == "sse":
        return f"{name}: the legacy SSE transport is not supported; use a streamable HTTP `url`"
    if kind not in ("stdio", "http"):
        return f"{name}: unknown transport {kind!r}"
    if kind == "stdio":
        if not str(raw.get("command") or "").strip():
            return f"{name}: missing `command`"
        if raw.get("url") or raw.get("headers"):
            return f"{name}: `url` and `headers` belong to the http transport"
        return ""
    if not str(raw.get("url") or "").strip():
        return f"{name}: missing `url`"
    if raw.get("command") or raw.get("args") or raw.get("env"):
        return f"{name}: `command`, `args` and `env` belong to the stdio transport"
    return ""


class Client:
    def __init__(self, server: Server) -> None:
        self.server = server
        self._proc: subprocess.Popen[str] | None = None
        self._inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [self.server.command, *self.server.args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, env={**os.environ, **self.server.env}, start_new_session=True,
            )
        except OSError as e:
            raise MCPError(f"cannot start {self.server.command!r}: {e}") from e
        threading.Thread(target=self._read_loop, daemon=True, name=f"mcp-{self.server.name}").start()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line or len(line) > LIMITS.mcp_message_bytes:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict) and message.get("id") is not None:
                self._inbox.put(message)

    def _send(self, message: dict[str, Any]) -> None:
        if self._proc is None or self._proc.poll() is not None or self._proc.stdin is None:
            raise MCPError("server is not running")
        try:
            self._proc.stdin.write(json.dumps(message) + "\n")
            self._proc.stdin.flush()
        except OSError as e:
            raise MCPError(f"write failed: {e}") from e

    def _await(self, message_id: int, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(f"timed out after {timeout_s:g}s")
            try:
                message = self._inbox.get(timeout=remaining)
            except queue.Empty:
                raise MCPError(f"timed out after {timeout_s:g}s") from None
            if message.get("id") != message_id:
                continue
            if message.get("error"):
                raise MCPError(str(message["error"].get("message") or message["error"]))
            return message.get("result") or {}

    def request(self, method: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            message_id = self._next_id
            self._send({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params})
            return self._await(message_id, timeout_s)

    def notify(self, method: str, params: dict[str, Any]) -> None:
        with self._lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def handshake(self, timeout_s: float) -> None:
        self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": CLIENT_NAME, "version": __version__},
        }, timeout_s)
        self.notify("notifications/initialized", {})

    def list_tools(self, timeout_s: float) -> list[dict[str, Any]]:
        found = self.request("tools/list", {}, timeout_s).get("tools")
        return [t for t in found or [] if isinstance(t, dict) and str(t.get("name") or "").strip()]

    def call(self, name: str, arguments: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments}, timeout_s)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=LIMITS.mcp_shutdown_wait_s)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()


def _sse_messages(body: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for event in body.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[len("data:"):].strip() for line in event.split("\n") if line.startswith("data:"))
        if not data:
            continue
        try:
            message = json.loads(data)
        except ValueError:
            continue
        if isinstance(message, dict):
            found.append(message)
    return found


class HttpClient:
    def __init__(self, server: Server) -> None:
        self.server = server
        self._http: httpx.Client | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._session_id = ""

    def start(self) -> None:
        self._http = httpx.Client(headers={**self.server.headers, "Content-Type": "application/json",
                                           "Accept": "application/json, text/event-stream"})

    def _post(self, message: dict[str, Any], timeout_s: float) -> httpx.Response:
        if self._http is None:
            raise MCPError("client is closed")
        headers = {"Mcp-Session-Id": self._session_id} if self._session_id else {}
        try:
            response = self._http.post(self.server.url, content=json.dumps(message).encode(), headers=headers, timeout=timeout_s)
        except httpx.HTTPError as e:
            raise MCPError(f"{type(e).__name__}: {e}") from e
        if session_id := response.headers.get("mcp-session-id", "").strip():
            self._session_id = session_id
        if response.status_code < 200 or response.status_code >= 300:
            raise MCPError(f"HTTP {response.status_code} from {self.server.url}: {response.text[:LIMITS.preview_error_chars]}")
        return response

    def request(self, method: str, params: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            message_id = self._next_id
        response = self._post({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params}, timeout_s)
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        candidates = _sse_messages(response.text) if content_type == "text/event-stream" else [response.json()]
        reply = next((m for m in candidates if isinstance(m, dict) and m.get("id") == message_id), None)
        if reply is None:
            raise MCPError(f"no response to {method} in the {content_type or 'empty'} body")
        if reply.get("error"):
            raise MCPError(str(reply["error"].get("message") or reply["error"]))
        return reply.get("result") or {}

    def notify(self, method: str, params: dict[str, Any], timeout_s: float = LIMITS.mcp_connect_timeout_s) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params}, timeout_s)

    def handshake(self, timeout_s: float) -> None:
        self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": CLIENT_NAME, "version": __version__},
        }, timeout_s)
        self.notify("notifications/initialized", {}, timeout_s)

    def list_tools(self, timeout_s: float) -> list[dict[str, Any]]:
        found = self.request("tools/list", {}, timeout_s).get("tools")
        return [t for t in found or [] if isinstance(t, dict) and str(t.get("name") or "").strip()]

    def call(self, name: str, arguments: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments}, timeout_s)

    def close(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            http.close()


def text_content(blocks: Any) -> tuple[str, int]:
    texts, dropped = [], 0
    for block in blocks or []:
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            texts.append(str(block["text"]))
        else:
            dropped += 1
    return "\n".join(texts), dropped


class RemoteTool(Tool):
    deferred = True

    def __init__(self, client: Client | HttpClient, remote: dict[str, Any]) -> None:
        self._client = client
        self._remote = str(remote["name"]).strip()
        self.server = client.server.name
        self.name = tool_name(self.server, self._remote)
        self.description = str(remote.get("description") or "").strip() or f"Call MCP tool {self.server}/{self._remote}"
        schema = remote.get("inputSchema")
        self.parameters = schema if isinstance(schema, dict) and schema.get("type") == "object" else EMPTY_SCHEMA
        self.safety = Safety(SideEffect.NETWORK, Permission.PROMPT,
                             f"Runs {self._remote} on the {self.server} MCP server over {client.server.transport}.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        try:
            payload = self._client.call(self._remote, args, LIMITS.mcp_call_timeout_s)
        except MCPError as e:
            return Result.error(f"Error: {self.name} failed: {e}")
        text, dropped = text_content(payload.get("content"))
        if dropped:
            text += f"\n[{dropped} non-text block(s) dropped; retrying will not return them as text]"
        text = text.strip() or "(empty result)"
        return Result.error(text) if payload.get("isError") else Result.success(text)


@dataclass
class Skipped:
    name: str
    error: str


@dataclass
class Bridge:
    clients: list[Client | HttpClient] = field(default_factory=list)
    tools: list[RemoteTool] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def close(self) -> None:
        for client in self.clients:
            client.close()
        self.clients.clear()


def _connect(server: Server, timeout_s: float) -> tuple[Client | HttpClient, list[dict[str, Any]]]:
    client: Client | HttpClient = HttpClient(server) if server.url else Client(server)
    client.start()
    try:
        client.handshake(timeout_s)
        return client, client.list_tools(timeout_s)
    except MCPError:
        client.close()
        raise


def connect_all(config: Config, registry: Registry, timeout_s: float = LIMITS.mcp_connect_timeout_s) -> Bridge:
    bridge = Bridge(problems=list(config.problems))
    if not config.servers:
        return bridge
    pool = ThreadPoolExecutor(max_workers=len(config.servers))
    try:
        futures = {pool.submit(_connect, server, timeout_s): server for server in config.servers}
        wait(list(futures), timeout=timeout_s)
        outcomes: dict[str, Any] = {}
        for future, server in futures.items():
            if not future.done():
                future.cancel()
                outcomes[server.name] = MCPError(f"connect timed out after {timeout_s:g}s")
                continue
            try:
                outcomes[server.name] = future.result()
            except (MCPError, OSError) as e:
                outcomes[server.name] = e
    finally:
        pool.shutdown(wait=False)

    staged: set[str] = set()
    for server in config.servers:
        outcome = outcomes[server.name]
        if isinstance(outcome, Exception):
            bridge.skipped.append(Skipped(server.name, str(outcome)))
            continue
        client, remotes = outcome
        tools = [RemoteTool(client, remote) for remote in remotes]
        names = [tool.name for tool in tools]
        clash = next((n for n in names if registry.get(n) is not None or n in staged or names.count(n) > 1), "")
        if clash:
            client.close()
            bridge.skipped.append(Skipped(server.name, f"tool {clash} conflicts with an already registered tool"))
            continue
        for tool in tools:
            registry.register(tool)
            staged.add(tool.name)
        bridge.clients.append(client)
        bridge.tools.extend(tools)
    return bridge
