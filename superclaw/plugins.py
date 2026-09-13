from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from superclaw.settings import PLUGIN_MANIFEST, PLUGIN_PARTS

_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_GIT = ("http://", "https://", "git@", "ssh://", "git://")


class PluginError(RuntimeError):
    pass


@dataclass(frozen=True)
class Plugin:
    id: str
    description: str
    version: str
    path: Path

    @property
    def parts(self) -> list[str]:
        return [part for part in PLUGIN_PARTS if (self.path / part).exists()]


def manifest(path: Path) -> Plugin:
    try:
        raw = json.loads((path / PLUGIN_MANIFEST).read_text())
    except (OSError, ValueError) as e:
        raise PluginError(f"{path / PLUGIN_MANIFEST}: {e}") from e
    plugin_id = str(raw.get("id") or "").strip() if isinstance(raw, dict) else ""
    if not _ID.match(plugin_id):
        raise PluginError(f"{path / PLUGIN_MANIFEST}: `id` must match [a-z0-9][a-z0-9._-]*, got {plugin_id!r}")
    return Plugin(plugin_id, str(raw.get("description") or ""), str(raw.get("version") or ""), path)


def load_plugins(roots: list[Path]) -> list[Plugin]:
    seen: dict[str, Plugin] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir(), key=lambda p: p.name):
            if not (entry / PLUGIN_MANIFEST).is_file():
                continue
            try:
                plugin = manifest(entry)
            except PluginError:
                continue
            seen.setdefault(plugin.id, plugin)
    return sorted(seen.values(), key=lambda p: p.id)


def locate(root: Path) -> Path:
    if (root / PLUGIN_MANIFEST).is_file():
        return root
    found = [p.parent for p in root.rglob(PLUGIN_MANIFEST) if ".git" not in p.parts]
    if len(found) != 1:
        raise PluginError(f"{root}: expected exactly one {PLUGIN_MANIFEST}, found {len(found)}")
    return found[0]


def install(source: str, root: Path) -> Plugin:
    source = source.strip()
    if not source:
        raise PluginError("a plugin source is required: a directory or a git URL")
    with tempfile.TemporaryDirectory(prefix="superclaw-plugin-") as scratch:
        if source.startswith(_GIT) or source.endswith(".git"):
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
        if target.exists():
            raise PluginError(f"plugin {plugin.id!r} is already installed at {target}; remove it first")
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(plugin_dir, target, ignore=shutil.ignore_patterns(".git"))
    return manifest(target)


def remove(plugin_id: str, root: Path) -> Path:
    if not _ID.match(plugin_id):
        raise PluginError(f"invalid plugin id {plugin_id!r}")
    target = root / plugin_id
    if not (target / PLUGIN_MANIFEST).is_file():
        raise PluginError(f"no plugin {plugin_id!r} under {root}")
    shutil.rmtree(target)
    return target
