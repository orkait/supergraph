from __future__ import annotations

import sys
from typing import IO, Any

from superclaw import __version__
from superclaw.app import Runtime
from superclaw.mcp import CLIENT_NAME, PROTOCOL_VERSION
from superclaw.rpc import INVALID_PARAMS, Conn, RpcError
from superclaw.settings import SERVED_TITLE, SERVED_TOOLS
from superclaw.tools import Tool, ToolContext


class Bridge:
    def __init__(self, rt: Runtime, conn: Conn) -> None:
        self.rt = rt
        self.conn = conn
        self.tools: list[Tool] = [t for t in (rt.registry.get(name) for name in SERVED_TOOLS) if t is not None]
        self._session = ""
        conn.handlers.update({"initialize": self.initialize, "tools/list": self.list_tools, "tools/call": self.call, "ping": self.ping})
        conn.notifiers["notifications/initialized"] = self.ignore

    def ping(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def ignore(self, params: dict[str, Any]) -> None:
        return None

    def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": str(params.get("protocolVersion") or PROTOCOL_VERSION),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": CLIENT_NAME, "version": __version__},
        }

    def list_tools(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"tools": [{"name": t.name, "description": t.description, "inputSchema": t.parameters} for t in self.tools]}

    def context(self) -> ToolContext:
        if not self._session and self.rt.store is not None:
            self._session = self.rt.store.create(cwd=str(self.rt.workspace), model="", title=SERVED_TITLE)
        return ToolContext(workspace=self.rt.workspace, session_id=self._session, extra_dirs=self.rt.extra_dirs)

    def call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name") or "")
        if not any(t.name == name for t in self.tools):
            raise RpcError(INVALID_PARAMS, f"unknown tool {name!r}; this server serves {', '.join(t.name for t in self.tools)}")
        res = self.rt.registry.run(name, dict(params.get("arguments") or {}), self.context())
        return {"content": [{"type": "text", "text": res.output}], "isError": not res.ok}


def serve(rt: Runtime, reader: IO[str] | None = None, writer: IO[str] | None = None) -> None:
    conn = Conn(reader or sys.stdin, writer or sys.stdout)
    Bridge(rt, conn)
    conn.serve()
