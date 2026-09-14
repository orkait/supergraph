from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from contextlib import suppress
from typing import Any

from superclaw.kernel import Kernel
from superclaw.runtime import approx_tokens
from superclaw.sandbox import Backend, Grant
from superclaw.settings import LIMITS
from superclaw.tools import Category, Permission, Result, Safety, SideEffect, Tool, ToolContext, jail

SANDBOX_MODES = ("use_default", "with_additional_permissions", "require_escalated")
_MS_PER_SECOND = 1000
_RUNNING = "running"
_EXITED = "exited"
_KILLED = "killed"
_TEST_RUNNER = re.compile(r"\b(pytest|py\.test|unittest|jest|vitest|mocha|go test|cargo test|npm test|pnpm test|yarn test|bun test|rspec|phpunit|mvn test|gradle test|ctest)\b")


class Job:
    def __init__(self, job_id: str, command: str, proc: subprocess.Popen[bytes]) -> None:
        self.id = job_id
        self.command = command
        self.proc = proc
        self.chunks: list[str] = []
        self.read = 0
        self.lock = threading.Lock()
        self.threads = [threading.Thread(target=self._drain, args=(stream,), daemon=True) for stream in (proc.stdout, proc.stderr)]
        for thread in self.threads:
            thread.start()

    def _drain(self, stream: Any) -> None:
        for line in iter(stream.readline, b""):
            text = line.decode("utf-8", errors="replace")
            with self.lock:
                if sum(len(c) for c in self.chunks) < LIMITS.shell_capture_bytes:
                    self.chunks.append(text)
        stream.close()

    @property
    def status(self) -> str:
        code = self.proc.poll()
        if code is None:
            return _RUNNING
        return _KILLED if code < 0 else _EXITED

    def drain(self) -> str:
        if self.proc.poll() is not None:
            for thread in self.threads:
                thread.join(LIMITS.shell_drain_timeout_s)
        with self.lock:
            fresh = "".join(self.chunks[self.read:])
            self.read = len(self.chunks)
        return fresh.rstrip("\n")

    def kill(self) -> None:
        if self.proc.poll() is None:
            with suppress(OSError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()

    def line(self) -> str:
        code = self.proc.poll()
        state = self.status if code is None else f"{self.status} {code}"
        return f"{self.id}  {state}  {self.command}"


class Jobs:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    def start(self, command: str, proc: subprocess.Popen[bytes]) -> Job:
        job = Job(f"bg_{len(self.jobs) + 1}", command, proc)
        self.jobs[job.id] = job
        return job

    def close(self) -> None:
        for job in self.jobs.values():
            job.kill()


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
            "capture": {"type": "string", "description": "Bind the full output in the python kernel under this name (a Run with .out, .code, .lines) instead of reading it all here."},
            "run_in_background": {"type": "boolean", "default": False,
                                  "description": "Return a job id at once and keep the command running. Use for anything slow (a test suite, a build, a server); read it with bash_output."},
        },
        "required": ["command", "description"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.SHELL, Permission.PROMPT, "Runs a shell command.")

    def __init__(self, backend: Backend | None = None, kernel: Kernel | None = None, jobs: Jobs | None = None) -> None:
        self.backend = backend
        self.kernel = kernel
        self.jobs = jobs if jobs is not None else Jobs()

    def category(self, args: dict[str, Any]) -> Category:
        return Category.TEST if _TEST_RUNNER.search(str(args.get("command") or "")) else Category.PROCESS

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        cwd = jail(ctx.roots, args.get("cwd") or ".")
        if not cwd.is_dir():
            return Result.error(f"Error: cwd is not a directory: {args.get('cwd')}")
        timeout = min(int(args.get("timeout_ms") or LIMITS.shell_timeout_ms), LIMITS.shell_max_timeout_ms) / _MS_PER_SECOND
        approval = ctx.state.get("approval") or {}
        argv = ["bash", "-c", args["command"]]
        if self.backend is not None and not approval.get("escalated"):
            extra = args.get("additional_permissions") or {}
            grant = Grant(paths=[*(extra.get("paths") or []), *(str(d) for d in ctx.extra_dirs)], network=bool(approval.get("network")))
            argv = self.backend.wrap(argv, cwd, ctx.workspace.resolve(), grant)
        env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PAGER": "cat", "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0"}
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
        if args.get("run_in_background"):
            job = self.jobs.start(str(args["command"]), proc)
            return Result.success(f"Started {job.id} in the background: {job.command}\nRead it with bash_output(id=\"{job.id}\"); it keeps running until it exits or you pass kill=true.")
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
        name = str(args.get("capture") or "")
        meta: dict[str, Any] = {}
        if name and self.kernel is not None and name.isidentifier():
            self.kernel.bind(name, out, proc.returncode)
            lines = out.splitlines()
            meta["full"] = out
            out = f"{name} = Run(code={proc.returncode}, {len(lines)} lines, {approx_tokens(out):,} tokens); tail:\n" + "\n".join(lines[-LIMITS.capture_preview_lines:])
        if proc.returncode != 0:
            out = f"{out}\n[exit {proc.returncode}]" if out else f"[exit {proc.returncode}]"
            return Result.error(out, meta=meta)
        return Result.success(out, meta=meta)


class BashOutput(Tool):
    name = "bash_output"
    deferred = True
    description = "Read new output from a background command started with bash(run_in_background=true), or stop it with kill=true."
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The job id bash returned. Omit to list every job."},
            "kill": {"type": "boolean", "default": False, "description": "Stop the command instead of reading it."},
        },
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.SHELL, Permission.ALLOW, "Reads or stops a background command already approved.")

    def __init__(self, jobs: Jobs) -> None:
        self.jobs = jobs

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        job_id = str(args.get("id") or "").strip()
        if not job_id:
            listing = "\n".join(job.line() for job in self.jobs.jobs.values())
            return Result.success(listing or "No background commands.")
        job = self.jobs.jobs.get(job_id)
        if job is None:
            known = ", ".join(self.jobs.jobs) or "none"
            return Result.error(f"Error: unknown job {job_id!r}; running or finished jobs: {known}")
        if args.get("kill"):
            job.kill()
            return Result.success(f"{job_id} killed.\n{job.drain()}".rstrip())
        fresh = job.drain()
        code = job.proc.poll()
        if code is None:
            return Result.success(f"{job_id} still running.\n{fresh}".rstrip() if fresh else f"{job_id} still running, no new output yet.")
        tail = f"{fresh}\n[exit {code}]" if fresh else f"[exit {code}]"
        return Result.success(tail) if code == 0 else Result.error(tail)
