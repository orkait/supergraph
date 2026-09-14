from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from superclaw.settings import LIMITS
from superclaw.tools.budget import Budget, Category, budget_output

_PROMPTS = Path(__file__).parent / "prompts"


class ReviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class Change:
    label: str
    patch: str
    untracked: list[str]


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise ReviewError(done.stderr.strip() or f"git {' '.join(args)} failed")
    return done.stdout


def uncommitted(cwd: Path) -> Change:
    untracked = [line for line in _git(cwd, "ls-files", "--others", "--exclude-standard").splitlines() if line]
    return Change("uncommitted changes against HEAD", _git(cwd, "diff", "HEAD"), untracked)


def against(cwd: Path, base: str) -> Change:
    return Change(f"changes on this branch since it left {base}", _git(cwd, "diff", f"{base}...HEAD"), [])


def commit(cwd: Path, sha: str) -> Change:
    subject = _git(cwd, "log", "-1", "--format=%h %s", sha).strip()
    parents = _git(cwd, "rev-list", "--parents", "-n", "1", sha).split()
    patch = _git(cwd, "diff", f"{sha}^", sha) if len(parents) > 1 else _git(cwd, "show", "--format=", sha)
    return Change(f"commit {subject}", patch, [])


def instruction() -> str:
    return (_PROMPTS / "review.md").read_text().strip()


def prompt(change: Change, extra: str = "") -> str:
    if not change.patch.strip() and not change.untracked:
        raise ReviewError(f"nothing to review: no {change.label}")
    budget = Budget(max_tokens=LIMITS.review_diff_tokens, max_chars=LIMITS.review_diff_tokens * LIMITS.chars_per_token)
    shown = budget_output(change.patch, Category.DIFF, budget)
    parts = [instruction()]
    if extra.strip():
        parts.append(f"<focus>\n{extra.strip()}\n</focus>")
    parts.append(f"<change scope=\"{change.label}\">\n{shown.text.strip()}\n</change>")
    if shown.truncated:
        parts.append(f"The diff was shortened from {shown.original_tokens:,} tokens to fit; read the files for anything cut.")
    if change.untracked:
        parts.append("New files not yet in git, read them if they matter: " + ", ".join(change.untracked))
    return "\n\n".join(parts)
