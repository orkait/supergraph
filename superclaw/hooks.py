from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVENTS = ("sessionStart", "beforeTool", "afterTool", "stop")
BLOCK_EXIT_CODE = 2
DEFAULT_TIMEOUT_S = 60


@dataclass(frozen=True)
class Hook:
    id: str
    event: str
    command: list[str]
    matcher: str = ""
    timeout_s: int = DEFAULT_TIMEOUT_S


@dataclass
class Outcome:
    blocked: bool = False
    blocked_by: str = ""
    context: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _parse(path: Path) -> list[Hook]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict) or not data.get("enabled"):
        return []
    hooks = []
    for raw in data.get("hooks") or []:
        if not isinstance(raw, dict) or raw.get("enabled", True) is False:
            continue
        command = raw.get("command")
        if isinstance(command, str):
            command = [command]
        if raw.get("event") not in EVENTS or not isinstance(command, list) or not command:
            continue
        hooks.append(Hook(
            id=str(raw.get("id") or f"{raw['event']}:{command[0]}"), event=str(raw["event"]),
            command=[str(c) for c in command], matcher=str(raw.get("matcher") or ""),
            timeout_s=int(raw.get("timeout_s") or DEFAULT_TIMEOUT_S),
        ))
    return hooks


def load_hooks(paths: list[Path]) -> list[Hook]:
    return [h for p in paths if p.is_file() for h in _parse(p)]


class Dispatcher:
    def __init__(self, hooks: list[Hook], cwd: Path) -> None:
        self.hooks = hooks
        self.cwd = cwd

    def _matches(self, hook: Hook, event: str, subject: str) -> bool:
        if hook.event != event:
            return False
        if not hook.matcher:
            return True
        try:
            return re.search(hook.matcher, subject) is not None
        except re.error:
            return False

    def dispatch(self, event: str, payload: dict[str, Any], subject: str = "") -> Outcome:
        outcome = Outcome()
        stdin = json.dumps({"event": event, **payload}).encode()
        for hook in self.hooks:
            if not self._matches(hook, event, subject):
                continue
            try:
                proc = subprocess.run(hook.command, input=stdin, cwd=self.cwd, capture_output=True, timeout=hook.timeout_s)
            except (OSError, subprocess.TimeoutExpired) as e:
                outcome.errors.append(f"{hook.id}: {type(e).__name__}: {e}")
                continue
            stdout = proc.stdout.decode("utf-8", errors="replace").strip()
            if stdout:
                try:
                    body = json.loads(stdout)
                    if isinstance(body, dict) and body.get("additionalContext"):
                        outcome.context.append(str(body["additionalContext"]))
                except ValueError:
                    outcome.errors.append(f"{hook.id}: stdout is not JSON")
            if proc.returncode == BLOCK_EXIT_CODE and event in ("beforeTool", "stop"):
                outcome.blocked = True
                outcome.blocked_by = hook.id
                reason = proc.stderr.decode("utf-8", errors="replace").strip()
                if reason:
                    outcome.context.append(reason)
                return outcome
            if proc.returncode not in (0, BLOCK_EXIT_CODE):
                outcome.errors.append(f"{hook.id}: exit {proc.returncode}: {proc.stderr.decode('utf-8', errors='replace').strip()[:200]}")
        return outcome
