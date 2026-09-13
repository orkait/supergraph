from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from superclaw import __version__
from superclaw.settings import LIMITS
from superclaw.tools import Permission, Registry, Result, Safety, SideEffect, Tool, ToolContext

PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "superclaw"
SERVER_KEYS = ("mcpServers", "servers")
TOOL_PREFIX = "mcp_"
EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": True}
_UNSAFE = re.compile(r"[^a-z0-9_]+")


class MCPError(RuntimeError):
    pass


@dataclass(frozen=True)
class Server:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


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


def load_config(paths: list[Path]) -> Config:
    config = Config()
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            config.problems.append(f"{path}: {type(e).__name__}: {e}")
            continue
        for name, raw in sorted(_servers_of(data).items()):
            if not isinstance(raw, dict) or raw.get("disabled"):
                continue
            if raw.get("url"):
                config.problems.append(f"{name}: only the stdio transport is supported; drop `url` or run the server locally")
                continue
            command = str(raw.get("command") or "").strip()
            if not command:
                config.problems.append(f"{name}: missing `command`")
                continue
            if name in seen:
                config.problems.append(f"{name}: already defined in an earlier config file; the first one wins")
                continue
            seen.add(name)
            config.servers.append(Server(
                name=name, command=command,
                args=[str(a) for a in raw.get("args") or []],
                env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            ))
    return config


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

    def __init__(self, client: Client, remote: dict[str, Any]) -> None:
        self._client = client
        self._remote = str(remote["name"]).strip()
        self.server = client.server.name
        self.name = tool_name(self.server, self._remote)
        self.description = str(remote.get("description") or "").strip() or f"Call MCP tool {self.server}/{self._remote}"
        schema = remote.get("inputSchema")
        self.parameters = schema if isinstance(schema, dict) and schema.get("type") == "object" else EMPTY_SCHEMA
        self.safety = Safety(SideEffect.NETWORK, Permission.PROMPT,
                             f"Runs {self._remote} on the {self.server} MCP server.")

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
    clients: list[Client] = field(default_factory=list)
    tools: list[RemoteTool] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def close(self) -> None:
        for client in self.clients:
            client.close()
        self.clients.clear()


def _connect(server: Server, timeout_s: float) -> tuple[Client, list[dict[str, Any]]]:
    client = Client(server)
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
