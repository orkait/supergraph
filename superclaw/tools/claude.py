from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from superclaw.settings import CLAUDE_CLI_BIN, CLAUDE_CLI_EDIT_MODE, CLAUDE_CLI_READ_TOOLS, CLAUDE_CLI_RESULT_TYPE, CLAUDE_CLI_SYSTEM_TYPE, LIMITS
from superclaw.text import clip, count
from superclaw.tools import Category, Permission, Result, Safety, SideEffect, Stopped, TimedOut, Tool, ToolContext, wait

which = shutil.which


def command(binary: str, task: str, args: dict[str, Any], extra_dirs: tuple[Path, ...]) -> list[str]:
    argv = [binary, "-p", task, "--output-format", "json", "--permission-prompts", "none", "--strict-mcp-config",
            "--max-budget-usd", f"{LIMITS.claude_cli_budget_usd:g}"]
    if args.get("allow_edits"):
        argv += ["--permission-mode", CLAUDE_CLI_EDIT_MODE]
    else:
        argv += ["--tools", ",".join(CLAUDE_CLI_READ_TOOLS)]
    if model := str(args.get("model") or "").strip():
        argv += ["--model", model]
    if session := str(args.get("session") or "").strip():
        argv += ["--resume", session]
    for directory in extra_dirs:
        argv += ["--add-dir", str(directory)]
    return argv


def report(payload: list[dict[str, Any]]) -> Result:
    final = next((m for m in reversed(payload) if m.get("type") == CLAUDE_CLI_RESULT_TYPE), {})
    model = next((str(m.get("model") or "") for m in payload if m.get("type") == CLAUDE_CLI_SYSTEM_TYPE), "")
    session = str(final.get("session_id") or "")
    answer = str(final.get("result") or "").strip()
    head = f"[claude {model or 'cli'}] {count(int(final.get('num_turns') or 0), 'turn')}, ${float(final.get('total_cost_usd') or 0.0):.4f}"
    if denied := len(final.get("permission_denials") or []):
        head += f", {count(denied, 'tool call')} denied"
    if session:
        head += f"\nresume it with session=\"{session}\""
    body = f"{head}\n\n{clip(answer, LIMITS.delegate_answer_tokens * LIMITS.chars_per_token)}"
    meta = {"full": answer, "session": session}
    if final.get("is_error") or not answer:
        return Result.error(f"{body}\n[claude did not finish: {final.get('subtype') or 'no result'}]", meta=meta)
    return Result.success(body, meta=meta)


class ClaudeCode(Tool):
    name = "claude"
    deferred = True
    description = (
        "Hand a self-contained task to Claude Code running headless here. "
        "It brings its own context, model and tools and runs on the Claude account signed in on this host, so use it for work you want kept out of your "
        "own context: a wide read, a review, a second opinion. "
        "Read-only unless allow_edits is true. You get its final answer and a §ref to the whole thing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Self-contained instructions; it knows nothing about this conversation."},
            "model": {"type": "string", "description": "Model alias (opus, sonnet, haiku) or full id; defaults to whatever that install is set to."},
            "allow_edits": {"type": "boolean", "default": False, "description": "Let it write files in this workspace instead of only reading."},
            "session": {"type": "string", "description": "Session id from an earlier call, to continue that conversation instead of starting one."},
        },
        "required": ["task"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.SHELL, Permission.PROMPT, "Runs a second coding agent with its own tools in this workspace.")
    output_category = Category.PROCESS

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        task = str(args.get("task") or "").strip()
        if not task:
            return Result.error("Error: task must not be empty")
        binary = which(CLAUDE_CLI_BIN)
        if binary is None:
            return Result.error(f"Error: {CLAUDE_CLI_BIN} is not installed; install Claude Code and sign in with `claude` first")
        proc = subprocess.Popen(command(binary, task, args, ctx.extra_dirs), cwd=ctx.workspace,
                                env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"},
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            stdout, stderr = wait(proc, LIMITS.claude_cli_timeout_s, ctx.cancelled)
        except Stopped:
            return Result.error("Error: stopped by the user")
        except TimedOut as e:
            return Result.error(f"Error: claude {e}")
        try:
            payload = json.loads(stdout.decode("utf-8", errors="replace"))
        except ValueError:
            detail = clip(stderr.decode("utf-8", errors="replace").strip() or "no output", LIMITS.preview_error_chars)
            return Result.error(f"Error: claude exited {proc.returncode} without a result: {detail}")
        return report(payload if isinstance(payload, list) else [payload])
