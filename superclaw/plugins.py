from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from superclaw.settings import (
    AGENTS_DIR,
    CLAUDE_HOOKS_FILE,
    CLAUDE_INSTALLED_FILE,
    CLAUDE_MCP_FILE,
    CLAUDE_PLUGIN_MANIFEST,
    COMMANDS_DIR,
    FORMAT_CLAUDE,
    FORMAT_SUPERCLAW,
    MCP_FILE,
    PLUGIN_MANIFEST,
    WORKSPACE_DIR,
    Settings,
)

_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_GIT = ("http://", "https://", "git@", "ssh://", "git://")
SKILLS_DIR = "skills"
HOOKS_FILE = "hooks.json"


class PluginError(RuntimeError):
    pass


@dataclass(frozen=True)
class Plugin:
    id: str
    description: str
    version: str
    path: Path
    format: str = FORMAT_SUPERCLAW
    skills: Path | None = None
    agents: Path | None = None
    commands: Path | None = None
    hooks: Path | None = None
    mcp: Path | None = None

    @property
    def parts(self) -> list[str]:
        return [name for name, found in (("skills", self.skills), ("agents", self.agents), ("commands", self.commands), ("hooks", self.hooks), ("mcp", self.mcp)) if found]


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise PluginError(f"{path}: {e}") from e
    return raw if isinstance(raw, dict) else {}


def _present(path: Path | None) -> Path | None:
    return path if path is not None and path.exists() else None


def _declared(root: Path, raw: dict[str, Any], key: str, default: str) -> Path | None:
    value = raw.get(key)
    if isinstance(value, str) and value.strip():
        return _present(root / value.strip().rstrip("/"))
    return _present(root / default)


def manifest(path: Path) -> Plugin:
    if (path / PLUGIN_MANIFEST).is_file():
        raw = _read(path / PLUGIN_MANIFEST)
        plugin_id = str(raw.get("id") or "").strip()
        if not _ID.match(plugin_id):
            raise PluginError(f"{path / PLUGIN_MANIFEST}: `id` must match [a-z0-9][a-z0-9._-]*, got {plugin_id!r}")
        return Plugin(plugin_id, str(raw.get("description") or ""), str(raw.get("version") or ""), path, FORMAT_SUPERCLAW,
                      _present(path / SKILLS_DIR), _present(path / AGENTS_DIR), _present(path / COMMANDS_DIR), _present(path / HOOKS_FILE), _present(path / MCP_FILE))
    if (path / CLAUDE_PLUGIN_MANIFEST).is_file():
        raw = _read(path / CLAUDE_PLUGIN_MANIFEST)
        plugin_id = str(raw.get("name") or "").strip().lower()
        if not _ID.match(plugin_id):
            raise PluginError(f"{path / CLAUDE_PLUGIN_MANIFEST}: `name` must match [a-z0-9][a-z0-9._-]*, got {plugin_id!r}")
        return Plugin(plugin_id, str(raw.get("description") or ""), str(raw.get("version") or ""), path, FORMAT_CLAUDE,
                      _declared(path, raw, "skills", SKILLS_DIR), _declared(path, raw, "agents", AGENTS_DIR), _declared(path, raw, "commands", COMMANDS_DIR),
                      _declared(path, raw, "hooks", CLAUDE_HOOKS_FILE), _declared(path, raw, "mcpServers", CLAUDE_MCP_FILE))
    raise PluginError(f"{path}: no {PLUGIN_MANIFEST} or {CLAUDE_PLUGIN_MANIFEST}")


def has_manifest(path: Path) -> bool:
    return (path / PLUGIN_MANIFEST).is_file() or (path / CLAUDE_PLUGIN_MANIFEST).is_file()


def load_plugins(roots: list[Path]) -> list[Plugin]:
    seen: dict[str, Plugin] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir(), key=lambda p: p.name):
            if not has_manifest(entry):
                continue
            try:
                plugin = manifest(entry)
            except PluginError:
                continue
            seen.setdefault(plugin.id, plugin)
    return sorted(seen.values(), key=lambda p: p.id)


def claude_installed(home: Path | None = None) -> list[Path]:
    registry = (home or Path.home()) / CLAUDE_INSTALLED_FILE
    if not registry.is_file():
        return []
    try:
        data = json.loads(registry.read_text())
    except (OSError, ValueError):
        return []
    paths = []
    for entries in (data.get("plugins") or {}).values() if isinstance(data, dict) else []:
        for entry in entries if isinstance(entries, list) else []:
            install_path = Path(str(entry.get("installPath") or "")) if isinstance(entry, dict) else None
            if install_path and has_manifest(install_path):
                paths.append(install_path)
    return paths


def discover(settings: Settings, workspace: Path | None = None, trusted: bool = True) -> list[Plugin]:
    roots = settings.plugin_roots(workspace) if trusted else [settings.user_plugins]
    found = load_plugins(roots)
    if settings.claude_plugins:
        known = {plugin.id for plugin in found}
        for path in claude_installed():
            try:
                plugin = manifest(path)
            except PluginError:
                continue
            if plugin.id not in known:
                found.append(plugin)
                known.add(plugin.id)
    return sorted(found, key=lambda p: p.id)


def locate(root: Path) -> Path:
    if has_manifest(root):
        return root
    found = sorted({p.parent for p in root.rglob(PLUGIN_MANIFEST) if ".git" not in p.parts}
                   | {p.parent.parent for p in root.rglob(CLAUDE_PLUGIN_MANIFEST) if ".git" not in p.parts})
    if len(found) != 1:
        raise PluginError(f"{root}: expected exactly one plugin manifest, found {len(found)}")
    return found[0]


def install(source: str, root: Path, link: bool = False) -> Plugin:
    source = source.strip()
    if not source:
        raise PluginError("a plugin source is required: a directory or a git URL")
    with tempfile.TemporaryDirectory(prefix="superclaw-plugin-") as scratch:
        if source.startswith(_GIT) or source.endswith(".git"):
            if link:
                raise PluginError("--link needs a local directory, not a git URL")
            done = subprocess.run(["git", "clone", "--depth", "1", "--quiet", source, scratch], capture_output=True, text=True)
            if done.returncode != 0:
                raise PluginError(done.stderr.strip() or f"git clone {source} failed")
            fetched = Path(scratch)
        else:
            fetched = Path(source).expanduser()
            if not fetched.is_dir():
                raise PluginError(f"not a directory: {source}")
        plugin_dir = locate(fetched)
        plugin = manifest(plugin_dir)
        target = root / plugin.id
        if target.exists() or target.is_symlink():
            raise PluginError(f"plugin {plugin.id!r} is already installed at {target}; remove it first")
        root.mkdir(parents=True, exist_ok=True)
        if link:
            target.symlink_to(plugin_dir.resolve(), target_is_directory=True)
        else:
            shutil.copytree(plugin_dir, target, ignore=shutil.ignore_patterns(".git"))
    return manifest(target)


def remove(plugin_id: str, root: Path) -> Path:
    if not _ID.match(plugin_id):
        raise PluginError(f"invalid plugin id {plugin_id!r}")
    target = root / plugin_id
    if not has_manifest(target):
        raise PluginError(f"no plugin {plugin_id!r} under {root}")
    if target.is_symlink():
        target.unlink()
    else:
        shutil.rmtree(target)
    return target


def workspace_plugins_dir(workspace: Path) -> Path:
    return Path(workspace) / WORKSPACE_DIR / "plugins"
