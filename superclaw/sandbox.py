from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

DENY_READ = ("~/.ssh", "~/.aws", "~/.gnupg", "~/.netrc", "~/.config/gh", "~/.docker/config.json", "~/.kube")
DENY_WRITE = (".env", ".git/hooks")


@dataclass
class Grant:
    paths: list[str] = field(default_factory=list)
    network: bool = False


class Backend(Protocol):
    name: str

    def wrap(self, argv: list[str], cwd: Path, workspace: Path, grant: Grant) -> list[str]: ...


class Bubblewrap:
    name = "bubblewrap"

    def wrap(self, argv: list[str], cwd: Path, workspace: Path, grant: Grant) -> list[str]:
        ws = str(workspace)
        cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--bind", ws, ws]
        for extra in grant.paths:
            cmd += ["--bind", extra, extra]
        for masked in DENY_READ:
            p = Path(masked).expanduser()
            if p.is_dir():
                cmd += ["--tmpfs", str(p)]
            elif p.is_file():
                cmd += ["--ro-bind", "/dev/null", str(p)]
        for protected in DENY_WRITE:
            p = workspace / protected
            if p.exists():
                cmd += ["--ro-bind", str(p), str(p)]
        cmd += ["--unshare-pid", "--die-with-parent", "--chdir", str(cwd)]
        if not grant.network:
            cmd.append("--unshare-net")
        return [*cmd, "--", *argv]


def detect() -> Backend | None:
    return Bubblewrap() if shutil.which("bwrap") else None
