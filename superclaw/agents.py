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


def _markdown(directory: Path) -> list[Path]:
    return [entry for entry in sorted(directory.iterdir(), key=lambda p: p.name) if entry.suffix == SUFFIX]


def _read(entry: Path, real_root: Path, *, require_name: bool) -> Agent | None:
    try:
        real = entry.resolve(strict=True)
    except OSError:
        return None
    if real_root not in real.parents or not real.is_file():
        return None
    fields, body = frontmatter(real.read_text(errors="replace"))
    if require_name and not fields.get("name"):
        return None
    return Agent(
        name=fields.get("name") or entry.stem,
        description=fields.get("description", ""),
        prompt=body.strip(),
        tools=_split(fields.get("tools", "")),
        model=fields.get("model", ""),
        path=str(entry),
    )


def _load_root(root: Path) -> list[Agent]:
    """Flat `<name>.md` files first, then one level of role directories.

    Claude-format plugins such as hyperstack keep a role as `agents/<role>/PROFILE.md`
    beside companion docs (`CHECKS.md`, `CONTEXT.md`) that carry no frontmatter, so a
    nested file counts only when its frontmatter names it. Flat files win a name clash.
    """
    if not root.is_dir():
        return []
    real_root = root.resolve()
    flat = [_read(entry, real_root, require_name=False) for entry in _markdown(root)]
    nested = [_read(entry, real_root, require_name=True)
              for directory in sorted(root.iterdir(), key=lambda p: p.name) if directory.is_dir()
              for entry in _markdown(directory)]
    return [agent for agent in flat + nested if agent is not None]


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
