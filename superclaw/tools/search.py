from __future__ import annotations

import json
from typing import Any

from superclaw.tools import Permission, Registry, Result, Safety, SideEffect, Tool, ToolContext

MAX_KEYWORD_MATCHES = 10


class ToolSearch(Tool):
    name = "tool_search"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": 'Either "select:name1,name2" for exact tool names, or keywords matched against names and descriptions.'}},
        "required": ["query"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "Loads already-registered tool schemas.")

    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    @property
    def description(self) -> str:
        lines = [f"- {t.name}: {t.summary()}" for t in self.registry.deferred()]
        return "Load the full schema of a deferred tool so you can call it next turn. Deferred tools:\n" + "\n".join(lines)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        query = str(args.get("query") or "").strip()
        deferred = {t.name: t for t in self.registry.deferred()}
        if query.startswith("select:"):
            wanted = [n.strip() for n in query[len("select:"):].split(",") if n.strip()]
            matches = [deferred[n] for n in wanted if n in deferred]
        else:
            words = [w.lower() for w in query.split()]
            matches = [t for t in deferred.values() if any(w in f"{t.name} {t.description}".lower() for w in words)][:MAX_KEYWORD_MATCHES]
        if not matches:
            return Result.error(f"Error: no deferred tool matches {query!r}. Available: {', '.join(deferred) or '(none)'}")
        rendered = "\n\n".join(json.dumps(t.definition()["function"], indent=1) for t in matches)
        return Result.success(f"Loaded for the next turn: {', '.join(t.name for t in matches)}\n\n{rendered}",
                              meta={"load_tools": [t.name for t in matches]})
