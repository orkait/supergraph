from __future__ import annotations

import hashlib
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from superclaw.settings import LIMITS

BRANCH_PREFIX = "superclaw/"
BUCKET_PREFIX = "superclaw-worktree-"
NAME_FORMAT = "task-%Y%m%d-%H%M%S"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SLUG = re.compile(r"[^a-z0-9._-]+")


class WorktreeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Worktree:
    name: str
    path: Path
    repo_root: Path
    branch: str
    reused: bool


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        raise WorktreeError(done.stderr.strip() or done.stdout.strip() or f"git {' '.join(args)} failed")
    return done.stdout.strip()


def _common_dir(cwd: Path) -> Path:
    return Path(_git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")).resolve()


def default_name() -> str:
    return time.strftime(NAME_FORMAT, time.gmtime())


def validate(name: str) -> str:
    if not _NAME.match(name) or len(name) > LIMITS.worktree_name_chars:
        raise WorktreeError(f"invalid worktree name {name!r}: letters, numbers, dots, dashes and underscores only")
    return name


def repo_key(repo_root: Path) -> str:
    digest = hashlib.sha256(str(repo_root).encode()).hexdigest()[: LIMITS.worktree_key_hash_chars]
    slug = _SLUG.sub("-", repo_root.name.lower()).strip("-._") or "repo"
    return f"{slug}-{digest}"


def prepare(cwd: Path, base_dir: Path, name: str = "") -> Worktree:
    name = validate(name.strip() or default_name())
    try:
        common = _common_dir(cwd)
    except WorktreeError as e:
        raise WorktreeError(f"not a git repository: {cwd}") from e
    repo_root = common.parent
    target = Path(base_dir).expanduser().resolve() / f"{BUCKET_PREFIX}{repo_key(repo_root)}" / name
    branch = f"{BRANCH_PREFIX}{name}"

    if target.exists():
        if not target.is_dir():
            raise WorktreeError(f"worktree path exists and is not a directory: {target}")
        if _common_dir(target) != common:
            raise WorktreeError(f"worktree path belongs to a different repository: {target}")
        return Worktree(name, target, repo_root, _git(target, "rev-parse", "--abbrev-ref", "HEAD"), reused=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    known = subprocess.run(["git", "rev-parse", "--verify", "--quiet", branch], cwd=repo_root,
                           capture_output=True, text=True).returncode == 0
    add = ["worktree", "add", str(target), branch] if known else ["worktree", "add", "-b", branch, str(target), "HEAD"]
    _git(repo_root, *add)
    return Worktree(name, target, repo_root, branch, reused=False)
