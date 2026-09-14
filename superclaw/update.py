from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse

import httpx

from superclaw.settings import LIMITS, PACKAGE, PYPI_URL, UV_TOOLS_MARKER


@dataclass(frozen=True)
class Install:
    method: str
    version: str
    source: str = ""


@dataclass(frozen=True)
class Plan:
    install: Install
    current: str
    latest: str
    command: list[str]

    @property
    def available(self) -> bool:
        return bool(self.latest) and self.latest != self.current


def detect(direct_url: str | None = None, version: str = "", executable: str = sys.executable) -> Install:
    if direct_url is None or not version:
        try:
            dist = metadata.distribution(PACKAGE)
            direct_url = direct_url if direct_url is not None else (dist.read_text("direct_url.json") or "")
            version = version or dist.version
        except metadata.PackageNotFoundError:
            return Install("unknown", version)
    try:
        info = json.loads(direct_url) if direct_url else {}
    except ValueError:
        info = {}
    if isinstance(info, dict) and (info.get("dir_info") or {}).get("editable"):
        return Install("editable", version, urlparse(str(info.get("url", ""))).path)
    if UV_TOOLS_MARKER in executable:
        return Install("uv-tool", version)
    return Install("pip", version)


def _git(path: str, *args: str) -> str:
    done = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else ""


def plan(install: Install, fetch: bool = True) -> Plan:
    if install.method == "editable":
        command = ["git", "-C", install.source, "pull", "--ff-only"]
        head = _git(install.source, "rev-parse", "--short", "HEAD")
        if not _git(install.source, "rev-parse", "--abbrev-ref", "@{upstream}"):
            return Plan(install, head, "", command)
        if fetch:
            _git(install.source, "fetch", "--quiet")
        behind = _git(install.source, "rev-list", "--count", "HEAD..@{upstream}")
        upstream = _git(install.source, "rev-parse", "--short", "@{upstream}") if behind not in ("", "0") else head
        return Plan(install, head, upstream, command)
    latest = ""
    if fetch and install.method != "unknown":
        try:
            latest = str(httpx.get(PYPI_URL, timeout=LIMITS.models_fetch_timeout_s).json()["info"]["version"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            latest = ""
    command = ["uv", "tool", "upgrade", PACKAGE] if install.method == "uv-tool" else [sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE]
    return Plan(install, install.version, latest, command)


def describe(found: Plan) -> list[str]:
    where = f" from {found.install.source}" if found.install.source else ""
    lines = [f"installed {found.install.method}{where} {found.current or found.install.version}"]
    if found.install.method == "unknown":
        lines.append(f"cannot tell how superclaw was installed; reinstall with `pip install -U {PACKAGE}` or `uv tool upgrade {PACKAGE}`")
    elif not found.latest and found.install.method == "editable":
        lines.append("no upstream branch to compare with; set one with `git branch --set-upstream-to`")
    elif not found.latest:
        lines.append("could not reach PyPI to check for a newer version")
    elif found.available:
        lines.append(f"update available: {found.current} -> {found.latest}; apply with `{' '.join(found.command)}`")
    else:
        lines.append("up to date")
    return lines


def apply(found: Plan) -> int:
    if not found.available:
        return 0
    done = subprocess.run(found.command, cwd=Path(found.install.source or "."), text=True, check=False)
    return done.returncode
