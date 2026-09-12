from __future__ import annotations

import os
import re
import signal
import subprocess
from typing import Any

from superclaw.sandbox import Backend, Grant
from superclaw.settings import LIMITS
from superclaw.tools import Category, Permission, Result, Safety, SideEffect, Tool, ToolContext, jail

SANDBOX_MODES = ("use_default", "with_additional_permissions", "require_escalated")
_MS_PER_SECOND = 1000
_TEST_RUNNER = re.compile(r"\b(pytest|py\.test|unittest|jest|vitest|mocha|go test|cargo test|npm test|pnpm test|yarn test|bun test|rspec|phpunit|mvn test|gradle test|ctest)\b")


class Bash(Tool):
    name = "bash"
    description = (
        "Run a shell command in the workspace sandbox (no network, writes only inside the workspace). "
        "For build, test, git and package commands; use the file tools to read and edit."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "description": {"type": "string", "description": "Why, one short line."},
            "cwd": {"type": "string", "description": "Relative to the workspace.", "default": "."},
            "timeout_ms": {"type": "integer", "default": LIMITS.shell_timeout_ms, "maximum": LIMITS.shell_max_timeout_ms},
            "sandbox_permissions": {"type": "string", "enum": list(SANDBOX_MODES), "default": "use_default",
                                    "description": "with_additional_permissions grants paths or network inside the sandbox; require_escalated runs on the host after approval."},
            "additional_permissions": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}, "network": {"type": "boolean"}}, "additionalProperties": False},
            "justification": {"type": "string", "description": "User-facing reason; required with escalation or extra permissions."},
            "prefix_rule": {"type": "array", "items": {"type": "string"}, "description": "Narrow prefix to remember if approved, e.g. [\"git\",\"pull\"]."},
        },
        "required": ["command", "description"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.SHELL, Permission.PROMPT, "Runs a shell command.")

    def __init__(self, backend: Backend | None = None) -> None:
        self.backend = backend

    def category(self, args: dict[str, Any]) -> Category:
        return Category.TEST if _TEST_RUNNER.search(str(args.get("command") or "")) else Category.PROCESS

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        cwd = jail(ctx.workspace, args.get("cwd") or ".")
        if not cwd.is_dir():
            return Result.error(f"Error: cwd is not a directory: {args.get('cwd')}")
        timeout = min(int(args.get("timeout_ms") or LIMITS.shell_timeout_ms), LIMITS.shell_max_timeout_ms) / _MS_PER_SECOND
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
        out = stdout[:LIMITS.shell_capture_bytes].decode("utf-8", errors="replace").rstrip("\n")
        err = stderr[:LIMITS.shell_capture_bytes].decode("utf-8", errors="replace").rstrip("\n")
        if err:
            out = f"{out}\n{err}" if out else err
        if proc.returncode != 0:
            out = f"{out}\n[exit {proc.returncode}]" if out else f"[exit {proc.returncode}]"
            return Result.error(out)
        return Result.success(out)
