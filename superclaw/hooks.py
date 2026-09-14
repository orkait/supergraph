from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from superclaw.settings import HOOK_SHELL, LIMITS, PLUGIN_ROOT_VARS

EVENTS = ("sessionStart", "userPrompt", "beforeTool", "afterTool", "stop")
CLAUDE_EVENTS = {"SessionStart": "sessionStart", "UserPromptSubmit": "userPrompt", "PreToolUse": "beforeTool", "PostToolUse": "afterTool", "Stop": "stop"}
CLAUDE_NAMES = {event: name for name, event in CLAUDE_EVENTS.items()}
BLOCKING_EVENTS = ("beforeTool", "stop")


@dataclass(frozen=True)
class Hook:
    id: str
    event: str
    command: list[str]
    matcher: str = ""
    timeout_s: int = LIMITS.hook_timeout_s
    root: Path | None = None


def substitute(text: str, root: Path | None) -> str:
    if root is None:
        return text
    for var in PLUGIN_ROOT_VARS:
        text = text.replace("${" + var + "}", str(root)).replace("$" + var, str(root))
    return text


def _parse_claude(groups: dict[str, Any], path: Path, root: Path | None) -> list[Hook]:
    hooks = []
    for claude_name, event in CLAUDE_EVENTS.items():
        for group in groups.get(claude_name) or []:
            if not isinstance(group, dict):
                continue
            for raw in group.get("hooks") or []:
                if not isinstance(raw, dict) or str(raw.get("type") or "command") != "command" or not str(raw.get("command") or "").strip():
                    continue
                command = substitute(str(raw["command"]).strip(), root)
                hooks.append(Hook(id=f"{path.parent.parent.name if root else path.stem}:{claude_name}:{len(hooks)}", event=event, command=[HOOK_SHELL, "-c", command],
                                  matcher=str(group.get("matcher") or ""), timeout_s=int(raw.get("timeout") or LIMITS.hook_timeout_s), root=root))
    return hooks


@dataclass
class Outcome:
    blocked: bool = False
    blocked_by: str = ""
    context: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _parse(path: Path, root: Path | None = None) -> list[Hook]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("hooks"), dict):
        return _parse_claude(data["hooks"], path, root)
    if not data.get("enabled"):
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
            command=[substitute(str(c), root) for c in command], matcher=str(raw.get("matcher") or ""),
            timeout_s=int(raw.get("timeout_s") or LIMITS.hook_timeout_s), root=root,
        ))
    return hooks


def load_hooks(entries: list[Path | tuple[Path, Path | None]]) -> list[Hook]:
    hooks = []
    for entry in entries:
        path, root = entry if isinstance(entry, tuple) else (entry, None)
        if path.is_file():
            hooks += _parse(path, root)
    return hooks


def _context_of(body: dict[str, Any]) -> str:
    specific = body.get("hookSpecificOutput")
    for candidate in (body.get("additionalContext"), specific.get("additionalContext") if isinstance(specific, dict) else None, body.get("systemMessage")):
        if candidate:
            return str(candidate)
    return ""


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
        stdin = json.dumps({"event": event, "hook_event_name": CLAUDE_NAMES.get(event, event), "session_id": payload.get("session", ""), "cwd": str(self.cwd),
                            "tool_name": payload.get("tool", ""), "tool_input": payload.get("args", {}), "source": subject, **payload}).encode()
        for hook in self.hooks:
            if not self._matches(hook, event, subject):
                continue
            env = {**os.environ, **({var: str(hook.root) for var in PLUGIN_ROOT_VARS} if hook.root else {})}
            try:
                proc = subprocess.run(hook.command, input=stdin, cwd=self.cwd, env=env, capture_output=True, timeout=hook.timeout_s)
            except (OSError, subprocess.TimeoutExpired) as e:
                outcome.errors.append(f"{hook.id}: {type(e).__name__}: {e}")
                continue
            stdout = proc.stdout.decode("utf-8", errors="replace").strip()
            decided_block = False
            if stdout:
                try:
                    body = json.loads(stdout)
                except ValueError:
                    outcome.errors.append(f"{hook.id}: stdout is not JSON")
                    body = {}
                if isinstance(body, dict):
                    if context := _context_of(body):
                        outcome.context.append(context)
                    decided_block = str(body.get("decision") or "").lower() == "block"
                    if decided_block and body.get("reason"):
                        outcome.context.append(str(body["reason"]))
            if (proc.returncode == LIMITS.hook_block_exit_code or decided_block) and event in BLOCKING_EVENTS:
                outcome.blocked = True
                outcome.blocked_by = hook.id
                reason = proc.stderr.decode("utf-8", errors="replace").strip()
                if reason:
                    outcome.context.append(reason)
                return outcome
            if proc.returncode not in (0, LIMITS.hook_block_exit_code):
                outcome.errors.append(f"{hook.id}: exit {proc.returncode}: {proc.stderr.decode('utf-8', errors='replace').strip()[:LIMITS.hook_error_chars]}")
        return outcome
