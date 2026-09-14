from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from superclaw.skills import Skill, frontmatter, load_skills

SUFFIX = ".md"
_PROMPTS = Path(__file__).parent / "prompts"
SKILL_TEMPLATE = "skill.md"
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_PLACEHOLDER = re.compile(r"\$(\$|ARGUMENTS|[1-9])")


@dataclass(frozen=True)
class UserCommand:
    name: str
    description: str
    template: str
    agent: str = ""
    model: str = ""
    path: str = ""
    skill: bool = False


def _load_root(root: Path) -> list[UserCommand]:
    if not root.is_dir():
        return []
    real_root = root.resolve()
    out: list[UserCommand] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        name = entry.stem.lower()
        if entry.suffix.lower() != SUFFIX or not _NAME.match(name):
            continue
        try:
            real = entry.resolve(strict=True)
        except OSError:
            continue
        if real_root not in real.parents or not real.is_file():
            continue
        fields, body = frontmatter(real.read_text(errors="replace"))
        out.append(UserCommand(
            name=name,
            description=fields.get("description") or f"User command: /{name}",
            template=body.strip(),
            agent=fields.get("agent") or fields.get("mode", ""),
            model=fields.get("model", ""),
            path=str(entry),
        ))
    return out


def from_skill(skill: Skill) -> UserCommand:
    template = (_PROMPTS / SKILL_TEMPLATE).read_text().replace("{name}", skill.name).replace("{body}", skill.content)
    return UserCommand(skill.name.lower(), skill.description or f"Skill: {skill.name}", template.strip(), path=skill.path, skill=True)


def load_commands(roots: list[Path], skill_roots: list[Path | tuple[Path, str]] | None = None) -> list[UserCommand]:
    seen: dict[str, UserCommand] = {}
    for root in roots:
        for command in _load_root(Path(root)):
            seen.setdefault(command.name, command)
    for skill in load_skills(skill_roots) if skill_roots else []:
        seen.setdefault(skill.name.lower(), from_skill(skill))
    return sorted(seen.values(), key=lambda c: c.name)


def expand(template: str, args: str) -> str:
    args = args.strip()
    try:
        positional = shlex.split(args)
    except ValueError:
        positional = args.split()
    if "$" not in template:
        return f"{template}\n\n{args}" if args else template

    def fill(match: re.Match[str]) -> str:
        placeholder = match.group(1)
        if placeholder == "$":
            return "$"
        if placeholder == "ARGUMENTS":
            return args
        index = int(placeholder) - 1
        return positional[index] if index < len(positional) else ""

    return _PLACEHOLDER.sub(fill, template)


def find(name: str, roots: list[Path], skill_roots: list[Path | tuple[Path, str]] | None = None) -> UserCommand | None:
    return next((c for c in load_commands(roots, skill_roots) if c.name == name.lower()), None)
