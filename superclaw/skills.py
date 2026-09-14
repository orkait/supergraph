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


def frontmatter(text: str) -> tuple[dict[str, str], str]:
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


def _load_root(root: Path, namespace: str = "") -> list[Skill]:
    if not root.is_dir():
        return []
    real_root = root.resolve()
    out: list[Skill] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        manifest = entry / SKILL_FILE
        if not manifest.exists():
            if not namespace and any((child / SKILL_FILE).exists() for child in entry.iterdir() if child.is_dir()):
                out += _load_root(entry.resolve() if entry.is_symlink() else entry, f"{entry.name}:")
            continue
        try:
            real = manifest.resolve(strict=True)
        except OSError:
            continue
        if real_root not in real.parents or not real.is_file():
            continue
        fields, body = frontmatter(real.read_text(errors="replace"))
        out.append(Skill(
            name=namespace + (fields.get("name") or entry.name),
            description=fields.get("description", ""),
            content=body.strip(),
            path=str(manifest),
        ))
    return out


def load_skills(roots: list[Path | tuple[Path, str]]) -> list[Skill]:
    seen: dict[str, Skill] = {}
    for root in roots:
        path, namespace = root if isinstance(root, tuple) else (root, "")
        for skill in _load_root(Path(path), f"{namespace}:" if namespace else ""):
            seen.setdefault(skill.name, skill)
    return sorted(seen.values(), key=lambda s: s.name)


def find_skill(skills: list[Skill], name: str) -> Skill | None:
    for skill in skills:
        if skill.name == name:
            return skill
    bare = [skill for skill in skills if skill.name.rpartition(":")[2] == name]
    return bare[0] if len(bare) == 1 else None
