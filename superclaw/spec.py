from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from superclaw.settings import LIMITS, PROMPTS_DIR, SPECS_DIR, WORKSPACE_DIR
from superclaw.tools import Display, Permission, Result, Safety, SideEffect, Tool, ToolContext

TOOL_NAME = "submit_spec"
CONTROL = "spec_review_required"
_SLUG = re.compile(r"[^a-z0-9]+")


class SpecError(RuntimeError):
    pass


@dataclass(frozen=True)
class Saved:
    id: str
    title: str
    path: Path
    relative: str


def specs_dir(workspace: Path) -> Path:
    return Path(workspace).resolve() / WORKSPACE_DIR / SPECS_DIR


def slug(title: str) -> str:
    return _SLUG.sub("-", title.lower()).strip("-")[: LIMITS.spec_slug_chars].strip("-") or "spec"


def save_draft(workspace: Path, title: str, plan: str) -> Saved:
    title, plan = title.strip(), plan.strip()
    if not title or not plan:
        raise SpecError("title and plan are both required")
    directory = specs_dir(workspace)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"{time.strftime('%Y-%m-%d', time.gmtime())}-{slug(title)}"
    for suffix in range(LIMITS.spec_collisions):
        spec_id = base if suffix == 0 else f"{base}-{suffix + 1}"
        path = directory / f"{spec_id}.md"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(f"# {title}\n\n{plan}\n")
        return Saved(spec_id, title, path, path.relative_to(Path(workspace).resolve()).as_posix())
    raise SpecError(f"too many specs named like {title!r} today")


def load(workspace: Path, ref: str) -> tuple[str, Path]:
    directory = specs_dir(workspace)
    candidate = Path(ref.strip())
    path = candidate if candidate.is_absolute() else (directory / f"{ref.strip()}.md" if "/" not in ref and not ref.endswith(".md") else Path(workspace).resolve() / candidate)
    path = path.resolve()
    if directory not in path.parents or not path.is_file():
        raise SpecError(f"no spec {ref!r} under {directory}")
    body = path.read_text(errors="replace").strip()
    if not body:
        raise SpecError(f"spec is empty: {path}")
    return body, path


def list_specs(workspace: Path) -> list[Path]:
    directory = specs_dir(workspace)
    return sorted(directory.glob("*.md")) if directory.is_dir() else []


def draft_prompt() -> str:
    return (PROMPTS_DIR / "spec.md").read_text().strip()


def implementation_prompt(body: str, path: Path, note: str = "") -> str:
    parts = ["Implement the following approved spec:"]
    if note.strip():
        parts.append(f"User note: {note.strip()}")
    parts += [body.strip(), f"Spec file: {path}",
              "If you need details that are not in the spec, inspect the workspace again. The spec is the approved source of truth for scope."]
    return "\n\n".join(parts)


class SubmitSpec(Tool):
    name = TOOL_NAME
    description = "Save the completed implementation spec under .superclaw/specs and stop for user review before implementation."
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short 3-6 word title for the spec."},
            "plan": {"type": "string", "description": "Complete markdown implementation spec."},
        },
        "required": ["title", "plan"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.WRITE, Permission.ALLOW, "Writes one markdown file under .superclaw/specs and ends the drafting run.")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        try:
            saved = save_draft(ctx.workspace, str(args.get("title") or ""), str(args.get("plan") or ""))
        except SpecError as e:
            return Result.error(f"Error: {e}")
        summary = f"Spec saved for review: {saved.relative}"
        return Result.success(summary, changed_files=[saved.relative], display=Display(summary=summary, kind="file"),
                              meta={"control": CONTROL, "spec_id": saved.id, "spec_path": str(saved.path)})
