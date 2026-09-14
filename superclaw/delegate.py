from __future__ import annotations

from typing import Any

from superclaw.settings import LIMITS
from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext

SPAWN_KEY = "delegate"


class Delegate(Tool):
    name = "delegate"
    deferred = True
    description = (
        "Run a sub-task in a fresh context. The child shares the workspace, tools and stored results but never your conversation: "
        "give it the task plus the §refs and file paths it needs. You get a short result and a §ref to its full answer; "
        "use it for reading or searching that would flood your own context."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Self-contained instructions; the child knows nothing else."},
            "agent": {"type": "string", "description": "Agent profile to run the child under; it narrows the child's prompt and tools."},
            "refs": {"type": "array", "items": {"type": "string"}, "description": "Stored results (§id) to hand over in full."},
            "files": {"type": "array", "items": {"type": "string"}, "description": "Workspace paths the child should start from."},
            "max_turns": {"type": "integer", "minimum": 1, "description": "Optional cap on the child's model turns; unset means the guards and budget end it."},
            "budget_tokens": {"type": "integer", "minimum": LIMITS.delegate_min_budget_tokens, "description": "Total tokens the child may spend; every turn costs its prompt again."},
        },
        "required": ["task"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "Runs a child agent whose own tool calls pass the same policy.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        spawn = ctx.state.get(SPAWN_KEY)
        if spawn is None:
            return Result.error("Error: delegation is not available in this run")
        return spawn(args)
