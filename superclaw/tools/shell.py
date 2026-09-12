from __future__ import annotations

import os
import signal
import subprocess
from typing import Any

from superclaw.sandbox import Backend, Grant
from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext, jail

DEFAULT_TIMEOUT_MS = 60_000
MAX_TIMEOUT_MS = 600_000
MAX_CAPTURE_BYTES = 1024 * 1024
SANDBOX_MODES = ("use_default", "with_additional_permissions", "require_escalated")


class Bash(Tool):
    name = "bash"
    description = (
        "Execute a shell command inside the workspace sandbox and return its output. "
        "Use it for build, test, git and package commands that have no native tool; "
        "prefer the file tools for reading and editing. The sandbox has no network and can only write "
        "inside the workspace; ask for more with sandbox_permissions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute with bash -c."},
            "description": {"type": "string", "description": "Why this command runs, one short active-voice line."},
            "cwd": {"type": "string", "description": "Directory to run in, relative to the workspace. Defaults to workspace root.", "default": "."},
            "timeout_ms": {"type": "integer", "description": "Command timeout in milliseconds.", "default": DEFAULT_TIMEOUT_MS, "minimum": 1, "maximum": MAX_TIMEOUT_MS},
            "sandbox_permissions": {
                "type": "string", "enum": list(SANDBOX_MODES), "default": "use_default",
                "description": "use_default runs sandboxed; with_additional_permissions adds the listed paths or network inside the sandbox; require_escalated runs outside the sandbox and needs approval with a justification.",
            },
            "additional_permissions": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Absolute directories to make writable."},
                    "network": {"type": "boolean", "description": "Allow network egress."},
                },
                "additionalProperties": False,
            },
            "justification": {"type": "string", "description": "User-facing reason, required with require_escalated or additional permissions."},
            "prefix_rule": {"type": "array", "items": {"type": "string"}, "description": "Narrow command prefix to remember if approved, for example [\"git\", \"pull\"]."},
        },
        "required": ["command", "description"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.SHELL, Permission.PROMPT, "Runs a shell command.")

    def __init__(self, backend: Backend | None = None) -> None:
        self.backend = backend

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        cwd = jail(ctx.workspace, args.get("cwd") or ".")
        if not cwd.is_dir():
            return Result.error(f"Error: cwd is not a directory: {args.get('cwd')}")
        timeout = min(int(args.get("timeout_ms") or DEFAULT_TIMEOUT_MS), MAX_TIMEOUT_MS) / 1000
        approval = ctx.state.get("approval") or {}
        argv = ["bash", "-c", args["command"]]
        if self.backend is not None and not approval.get("escalated"):
            extra = args.get("additional_permissions") or {}
            grant = Grant(paths=list(extra.get("paths") or []), network=bool(approval.get("network")))
            argv = self.backend.wrap(argv, cwd, ctx.workspace.resolve(), grant)
        env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PAGER": "cat", "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0"}
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return Result.error(f"Error: command timed out after {timeout:g}s")
        out = stdout[:MAX_CAPTURE_BYTES].decode("utf-8", errors="replace").rstrip("\n")
        err = stderr[:MAX_CAPTURE_BYTES].decode("utf-8", errors="replace").rstrip("\n")
        if err:
            out = f"{out}\n{err}" if out else err
        if proc.returncode != 0:
            out = f"{out}\n[exit {proc.returncode}]" if out else f"[exit {proc.returncode}]"
            return Result.error(out)
        return Result.success(out)
