from __future__ import annotations

from pathlib import Path
from typing import Any

from superclaw.skills import find_skill, load_skills, render_skill
from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext


def _from_workspace(path: str, workspace: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(workspace).resolve())
    except (OSError, ValueError):
        return False
    return True


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

    def __init__(self, roots: list[Path | tuple[Path, str]]) -> None:
        self.roots = roots

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        name = str(args.get("name") or "").strip()
        skills = load_skills(self.roots)
        if (skill := find_skill(skills, name)) is None:
            names = ", ".join(s.name for s in skills) or "(none installed)"
            return Result.error(f"Error: unknown skill {name!r}. Available skills: {names}.")
        if _from_workspace(skill.path, ctx.workspace):
            return Result.success(skill.content)
        return Result.success(render_skill(skill, ""), meta={"trusted": True})
