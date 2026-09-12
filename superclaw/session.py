from __future__ import annotations

import hashlib
import json
import secrets
import time
from typing import Any

from supergraph.core.errors import SuperGraphError

from superclaw.compaction import SUMMARY_LABEL
from superclaw.runtime import Message, ToolCall

NAMESPACE = "superclaw"


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _lit(value: Any) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


class SessionStore:
    def __init__(self, gs: Any) -> None:
        self._gs = gs

    def _x(self, query: str):
        return self._gs.execute(query, namespace=NAMESPACE)

    def create(self, *, cwd: str, model: str, title: str = "", parent: str = "") -> str:
        sid = f"s_{time.strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(3)}"
        self._x(
            f'CREATE NODE {_lit("session:" + sid)} kind = "session" sid = {_lit(sid)} cwd = {_lit(cwd)} '
            f'model = {_lit(model)} title = {_lit(title)} parent = {_lit(parent)} '
            f'created = {time.time_ns()} event_count = 0'
        )
        return sid

    def get(self, sid: str) -> dict[str, Any] | None:
        try:
            row = self._x(f'NODE {_lit("session:" + sid)}').data
        except SuperGraphError:
            return None
        if not row:
            return None
        return {
            "id": row["sid"], "cwd": row.get("cwd", ""), "model": row.get("model", ""),
            "title": row.get("title", ""), "parent": row.get("parent", ""),
            "created": row.get("created", 0), "event_count": row.get("event_count", 0),
        }

    def list(self) -> list[dict[str, Any]]:
        rows = self._x('NODES WHERE kind = "session" LIMIT 1000').data or []
        rows.sort(key=lambda r: (r.get("created", 0), r.get("sid", "")), reverse=True)
        return [
            {"id": r["sid"], "cwd": r.get("cwd", ""), "model": r.get("model", ""), "title": r.get("title", ""),
             "parent": r.get("parent", ""), "created": r.get("created", 0), "event_count": r.get("event_count", 0)}
            for r in rows
        ]

    def latest(self) -> str | None:
        rows = self.list()
        return rows[0]["id"] if rows else None

    def append(self, sid: str, etype: str, payload: dict[str, Any]) -> int:
        meta = self.get(sid)
        if meta is None:
            raise KeyError(f"unknown session {sid}")
        seq = int(meta["event_count"]) + 1
        node = f"ev:{sid}:{seq:06d}"
        self._x(
            f'CREATE NODE {_lit(node)} kind = "event" sid = {_lit(sid)} seq = {seq} etype = {_lit(etype)} '
            f'DOCUMENT {_lit(json.dumps(payload))}'
        )
        self._x(f'CREATE EDGE {_lit("session:" + sid)} -> {_lit(node)} kind = "has_event"')
        self._x(f'UPDATE NODE {_lit("session:" + sid)} SET event_count = {seq}')
        return seq

    def events(self, sid: str) -> list[dict[str, Any]]:
        rows = self._x(f'NODES WHERE kind = "event" AND sid = {_lit(sid)} ORDER BY seq ASC LIMIT 100000').data or []
        out = []
        for r in rows:
            doc = self._x(f'NODE {_lit(r["id"])} WITH DOCUMENT').data.get("_document") or "{}"
            out.append({"seq": r["seq"], "type": r["etype"], "payload": json.loads(doc)})
        return out

    def fork(self, sid: str) -> str:
        meta = self.get(sid)
        if meta is None:
            raise KeyError(f"unknown session {sid}")
        new = self.create(cwd=meta["cwd"], model=meta["model"], title=meta["title"], parent=sid)
        for ev in self.events(sid):
            self.append(new, ev["type"], ev["payload"])
        self._x(f'CREATE EDGE {_lit("session:" + new)} -> {_lit("session:" + sid)} kind = "forked_from"')
        return new

    def last_prompt(self, sid: str) -> dict[str, Any] | None:
        prompts = [ev["payload"] for ev in self.events(sid) if ev["type"] == "prompt"]
        return prompts[-1] if prompts else None

    def plan(self, sid: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for ev in self.events(sid):
            if ev["type"] == "plan":
                items = ev["payload"].get("items", [])
        return items

    def replay(self, sid: str) -> list[Message]:
        timeline: list[tuple[int, Message]] = []
        for ev in self.events(sid):
            seq, p = ev["seq"], ev["payload"]
            if ev["type"] == "message":
                timeline.append((seq, Message(
                    role=p["role"], content=p.get("content", ""),
                    tool_calls=[ToolCall(c["id"], c["name"], c["arguments"]) for c in p.get("tool_calls", [])],
                )))
            elif ev["type"] == "tool_result":
                timeline.append((seq, Message(
                    role="tool", content=p.get("output", ""), tool_call_id=p["tool_call_id"], is_error=not p.get("ok", True),
                )))
            elif ev["type"] == "compaction":
                through = int(p.get("through_seq", 0))
                kept = [(s, m) for s, m in timeline if s > through]
                timeline = [(seq, Message(role="user", content=f"{SUMMARY_LABEL}\n{p.get('summary', '')}")), *kept]
        return [m for _, m in timeline]
