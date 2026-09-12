from __future__ import annotations

from pathlib import Path
from typing import Any

from superclaw.skills import load_skills
from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext


class SkillTool(Tool):
    name = "skill"
    description = "Load a named skill and return its full instructions. Call it before acting on a request that matches a listed skill."
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "The name of the skill to load."}},
        "required": ["name"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.READ, Permission.ALLOW, "Reads a skill file.")

    def __init__(self, roots: list[Path]) -> None:
        self.roots = roots

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        name = str(args.get("name") or "").strip()
        skills = load_skills(self.roots)
        for skill in skills:
            if skill.name == name:
                return Result.success(skill.content)
        names = ", ".join(s.name for s in skills) or "(none installed)"
        return Result.error(f"Error: unknown skill {name!r}. Available skills: {names}.")
