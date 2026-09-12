from __future__ import annotations

from typing import Any

from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext

STATUSES = ("pending", "in_progress", "completed", "failed")
_COERCE = {
    "done": "completed", "complete": "completed", "completed": "completed", "finished": "completed",
    "todo": "pending", "pending": "pending", "open": "pending", "not_started": "pending",
    "wip": "in_progress", "in_progress": "in_progress", "in-progress": "in_progress", "doing": "in_progress", "active": "in_progress",
    "failed": "failed", "error": "failed", "blocked": "failed",
}


def parse_plan(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        raise ValueError("plan must be an array")
    items: list[dict[str, str]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or not str(entry.get("content") or "").strip():
            raise ValueError(f"plan item {i + 1} needs a non-empty content")
        status = _COERCE.get(str(entry.get("status") or "pending").strip().lower(), "pending")
        item = {"content": str(entry["content"]).strip(), "status": status}
        if entry.get("notes"):
            item["notes"] = str(entry["notes"]).strip()
        items.append(item)
    last_in_progress = None
    for i, item in enumerate(items):
        if item["status"] == "in_progress":
            if last_in_progress is not None:
                items[last_in_progress]["status"] = "completed"
            last_in_progress = i
    return items


def format_plan(items: list[dict[str, str]]) -> str:
    if not items:
        return "Plan is currently empty."
    lines = []
    for i, item in enumerate(items, 1):
        line = f"{i}. [{item['status']}] {item['content']}"
        if item.get("notes"):
            line += f"\n   Notes: {item['notes']}"
        lines.append(line)
    return "Current Plan:\n" + "\n".join(lines)


def pending_items(state: dict[str, Any]) -> list[str]:
    return [i["content"] for i in state.get("plan", []) if i["status"] in ("pending", "in_progress")]


class UpdatePlan(Tool):
    name = "update_plan"
    description = (
        "Create or update the plan for work spanning multiple components or many tool calls. "
        "Skip it for bounded changes, lookups and explanations. Pass the full ordered list each call; "
        "it replaces the previous plan. Keep at most one item in_progress."
    )
    parameters = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "description": "Ordered list of plan items, replacing any previous plan.",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The plan step description."},
                        "status": {"type": "string", "enum": list(STATUSES), "description": "Status of this step."},
                        "notes": {"type": "string", "description": "Optional notes for this step."},
                    },
                    "required": ["content"],
                },
            },
        },
        "required": ["plan"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.NONE, Permission.ALLOW, "Updates planning state only.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        try:
            items = parse_plan(args.get("plan"))
        except ValueError as e:
            return Result.error(f"Error: invalid arguments for update_plan: {e}")
        ctx.state["plan"] = items
        return Result.success(format_plan(items))
