from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SKILL_FILE = "SKILL.md"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    path: str


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fields: dict[str, str] = {}
            for line in lines[1:i]:
                key, sep, value = line.partition(":")
                if sep:
                    fields[key.strip().lower()] = value.strip().strip("\"'")
            return fields, "\n".join(lines[i + 1:])
    return {}, text


def _load_root(root: Path) -> list[Skill]:
    if not root.is_dir():
        return []
    real_root = root.resolve()
    out: list[Skill] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        manifest = entry / SKILL_FILE
        try:
            real = manifest.resolve(strict=True)
        except OSError:
            continue
        if real_root not in real.parents or not real.is_file():
            continue
        fields, body = _frontmatter(real.read_text(errors="replace"))
        out.append(Skill(
            name=fields.get("name") or entry.name,
            description=fields.get("description", ""),
            content=body.strip(),
            path=str(manifest),
        ))
    return out


def load_skills(roots: list[Path]) -> list[Skill]:
    seen: dict[str, Skill] = {}
    for root in roots:
        for skill in _load_root(Path(root)):
            seen.setdefault(skill.name, skill)
    return sorted(seen.values(), key=lambda s: s.name)
