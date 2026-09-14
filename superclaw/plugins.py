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
    GITHUB_URL,
    MARKETPLACE_MANIFEST,
    MARKETPLACE_SEP,
    MCP_FILE,
    PLUGIN_MANIFEST,
    WORKSPACE_DIR,
    Settings,
)

_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_GIT = ("http://", "https://", "git@", "ssh://", "git://")
_SHA = re.compile(r"^[0-9a-f]{7,40}$")
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
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


def claude_installed(claude_dir: Path) -> list[Path]:
    registry = claude_dir / CLAUDE_INSTALLED_FILE
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
    if settings.claude_config:
        known = {plugin.id for plugin in found}
        for path in claude_installed(settings.claude_dir):
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


def is_git(source: str) -> bool:
    return source.startswith(_GIT) or source.endswith(".git")


def clone(source: str, scratch: str, ref: str = "") -> Path:
    sha = bool(_SHA.match(ref))
    command = ["git", "clone", "--quiet", *([] if sha else ["--depth", "1"]), *(["--branch", ref] if ref and not sha else []), source, scratch]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode == 0 and sha:
        done = subprocess.run(["git", "-C", scratch, "checkout", "--quiet", ref], capture_output=True, text=True)
    if done.returncode != 0:
        raise PluginError(done.stderr.strip() or f"git clone {source} failed")
    return Path(scratch)


def fetch(source: str, scratch: str, link: bool = False, ref: str = "", subdir: str = "") -> Path:
    if is_git(source):
        if link:
            raise PluginError("--link needs a local directory, not a git URL")
        fetched = clone(source, scratch, ref)
    else:
        fetched = Path(source).expanduser()
    fetched = fetched / subdir if subdir else fetched
    if not fetched.is_dir():
        raise PluginError(f"not a directory: {fetched}")
    return fetched


def install(source: str, root: Path, link: bool = False, ref: str = "", subdir: str = "") -> Plugin:
    source = source.strip()
    if not source:
        raise PluginError("a plugin source is required: a directory, a git URL, or <plugin>@<marketplace>")
    with tempfile.TemporaryDirectory(prefix="superclaw-plugin-") as scratch:
        plugin_dir = locate(fetch(source, scratch, link, ref, subdir))
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


@dataclass(frozen=True)
class Marketplace:
    name: str
    description: str
    path: Path
    plugins: dict[str, Any]


def read_marketplace(path: Path) -> Marketplace:
    raw = _read(path / MARKETPLACE_MANIFEST) if (path / MARKETPLACE_MANIFEST).is_file() else None
    if raw is None:
        raise PluginError(f"{path}: no {MARKETPLACE_MANIFEST}")
    name = str(raw.get("name") or "").strip().lower()
    if not _ID.match(name):
        raise PluginError(f"{path / MARKETPLACE_MANIFEST}: `name` must match [a-z0-9][a-z0-9._-]*, got {name!r}")
    entries = {str(p.get("name") or "").strip().lower(): p for p in raw.get("plugins") or [] if isinstance(p, dict) and p.get("name")}
    return Marketplace(name, str(raw.get("description") or (raw.get("metadata") or {}).get("description") or ""), path, entries)


def load_marketplaces(root: Path) -> list[Marketplace]:
    found = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name) if root.is_dir() else []:
        try:
            found.append(read_marketplace(entry))
        except PluginError:
            continue
    return found


def add_marketplace(source: str, root: Path, link: bool = False) -> Marketplace:
    source = source.strip()
    if _REPO.match(source) and not Path(source).expanduser().is_dir():
        source = GITHUB_URL.format(repo=source)
    with tempfile.TemporaryDirectory(prefix="superclaw-marketplace-") as scratch:
        fetched = fetch(source, scratch, link)
        market = read_marketplace(fetched)
        target = root / market.name
        if target.exists() or target.is_symlink():
            raise PluginError(f"marketplace {market.name!r} already exists at {target}; remove it first")
        root.mkdir(parents=True, exist_ok=True)
        if link:
            target.symlink_to(fetched.resolve(), target_is_directory=True)
        else:
            shutil.copytree(fetched, target, ignore=shutil.ignore_patterns(".git"))
    return read_marketplace(target)


def remove_marketplace(name: str, root: Path) -> Path:
    target = root / name
    if not _ID.match(name) or not (target / MARKETPLACE_MANIFEST).is_file():
        raise PluginError(f"no marketplace {name!r} under {root}")
    if target.is_symlink():
        target.unlink()
    else:
        shutil.rmtree(target)
    return target


def resolve_source(market: Marketplace, plugin: str) -> tuple[str, str, str]:
    entry = market.plugins.get(plugin.strip().lower())
    if entry is None:
        raise PluginError(f"marketplace {market.name!r} has no plugin {plugin!r}; it offers {', '.join(sorted(market.plugins)) or 'nothing'}")
    source = entry.get("source")
    if isinstance(source, str):
        return str((market.path / source).resolve()) if not is_git(source) else source, "", ""
    if isinstance(source, dict):
        kind = str(source.get("source") or "")
        if kind == "url" and source.get("url"):
            return str(source["url"]), "", ""
        if kind == "github" and source.get("repo"):
            return GITHUB_URL.format(repo=source["repo"]), "", ""
        if kind == "git-subdir" and source.get("url") and source.get("path"):
            return str(source["url"]), str(source.get("ref") or ""), str(source["path"]).strip("/")
    raise PluginError(f"marketplace {market.name!r}: plugin {plugin!r} has an unsupported source {source!r}")


def is_marketplace_ref(source: str) -> bool:
    return MARKETPLACE_SEP in source and not is_git(source) and not Path(source).expanduser().exists()


def install_from_marketplace(spec: str, marketplaces: Path, root: Path) -> Plugin:
    plugin, _, market_name = spec.partition(MARKETPLACE_SEP)
    market = next((m for m in load_marketplaces(marketplaces) if m.name == market_name.strip().lower()), None)
    if market is None:
        raise PluginError(f"no marketplace {market_name!r}; add one with `superclaw plugin marketplace add <owner/repo|url|dir>`")
    source, ref, subdir = resolve_source(market, plugin)
    return install(source, root, ref=ref, subdir=subdir)


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
