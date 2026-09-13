from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from superclaw.skills import frontmatter

SUFFIX = ".md"


@dataclass(frozen=True)
class Agent:
    name: str
    description: str
    prompt: str
    tools: frozenset[str] = frozenset()
    model: str = ""
    path: str = ""


def _split(value: str) -> frozenset[str]:
    return frozenset(part for part in (p.strip() for p in value.replace(",", " ").split()) if part)


def _load_root(root: Path) -> list[Agent]:
    if not root.is_dir():
        return []
    real_root = root.resolve()
    out: list[Agent] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.suffix != SUFFIX:
            continue
        try:
            real = entry.resolve(strict=True)
        except OSError:
            continue
        if real_root not in real.parents or not real.is_file():
            continue
        fields, body = frontmatter(real.read_text(errors="replace"))
        out.append(Agent(
            name=fields.get("name") or entry.stem,
            description=fields.get("description", ""),
            prompt=body.strip(),
            tools=_split(fields.get("tools", "")),
            model=fields.get("model", ""),
            path=str(entry),
        ))
    return out


def load_agents(roots: list[Path]) -> list[Agent]:
    seen: dict[str, Agent] = {}
    for root in roots:
        for agent in _load_root(Path(root)):
            seen.setdefault(agent.name, agent)
    return sorted(seen.values(), key=lambda a: a.name)


def resolve(name: str, roots: list[Path]) -> Agent:
    agents = load_agents(roots)
    found = next((a for a in agents if a.name == name), None)
    if found is None:
        known = ", ".join(a.name for a in agents) or "none defined"
        raise KeyError(f"unknown agent {name!r}; available: {known}")
    return found
