from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from superclaw.sandbox import Backend, Grant
from superclaw.settings import LIMITS
from superclaw.tools import Category, Permission, Result, Safety, SideEffect, Tool, ToolContext

READ_VERBS = ("NODE", "NODES", "EDGES", "TRAVERSE", "SUBGRAPH", "PATH", "PATHS", "SHORTEST", "DISTANCE", "ANCESTORS", "DESCENDANTS",
              "COMMON", "MATCH", "COUNT", "AGGREGATE", "RECALL", "SIMILAR", "LEXICAL", "REMEMBER", "ANSWER")

CHILD = r'''
import base64, contextlib, io, json, pickle, subprocess, sys, traceback

class Run:
    def __init__(self, out, code):
        self.out, self.code = out, code
    def __reduce__(self):
        return (Run, (self.out, self.code))
    @property
    def lines(self):
        return self.out.splitlines()
    def __repr__(self):
        return f"Run(code={self.code}, {len(self.out)} chars)"

_pipe = sys.__stdout__

def _rpc(kind, **payload):
    _pipe.write(json.dumps({"rpc": kind, **payload}) + "\n"); _pipe.flush()
    reply = json.loads(sys.__stdin__.readline())
    if reply.get("error"):
        raise RuntimeError(reply["error"])
    return reply["value"]

def obs(ref):
    return _rpc("obs", ref=str(ref).lstrip("§"))

def query(dsl):
    return _rpc("query", dsl=dsl)

def sh(command, timeout=__TIMEOUT__):
    p = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
    return Run((p.stdout + p.stderr).rstrip("\n"), p.returncode)

ns = {"obs": obs, "query": query, "sh": sh, "Run": Run}
_builtin = set(ns)

def _dump():
    keep = {}
    for key, value in ns.items():
        if key in _builtin or key.startswith("__"):
            continue
        try:
            pickle.dumps(value); keep[key] = value
        except Exception:
            pass
    return base64.b64encode(pickle.dumps(keep)).decode()

for line in sys.stdin:
    req = json.loads(line)
    if "bind" in req:
        ns[req["bind"]] = Run(**req["value"]); continue
    if "dump" in req:
        _pipe.write(json.dumps({"ok": True, "out": _dump()}) + "\n"); _pipe.flush(); continue
    if "load" in req:
        ns.update(pickle.loads(base64.b64decode(req["load"]))); continue
    buf = io.StringIO()
    ok = True
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(req["code"], "<python>", "exec"), ns)
        except BaseException:
            ok = False
            traceback.print_exc(limit=-__TRACE__)
    _pipe.write(json.dumps({"ok": ok, "out": buf.getvalue()}) + "\n"); _pipe.flush()
'''


class Kernel:
    def __init__(self, workspace: Path, backend: Backend | None, resolve: Callable[[str, dict[str, Any]], Any]) -> None:
        self.workspace = workspace.resolve()
        self.backend = backend
        self.resolve = resolve
        self._proc: subprocess.Popen[str] | None = None

    def _start(self) -> subprocess.Popen[str]:
        source = CHILD.replace("__TIMEOUT__", str(LIMITS.shell_max_timeout_ms // _MS_PER_SECOND)).replace("__TRACE__", str(LIMITS.kernel_trace_depth))
        argv = [sys.executable, "-u", "-c", source]
        if self.backend is not None:
            argv = self.backend.wrap(argv, self.workspace, self.workspace, Grant())
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "NO_COLOR": "1"}
        self._proc = subprocess.Popen(argv, cwd=self.workspace, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, start_new_session=True)
        return self._proc

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _send(self, message: dict[str, Any]) -> None:
        proc = self._proc if self.alive else self._start()
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    def _serve(self, timeout_s: float) -> dict[str, Any]:
        proc = self._proc
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise BrokenPipeError("kernel exited")
                reply = json.loads(line)
                if "rpc" not in reply:
                    return reply
                try:
                    answer = {"value": self.resolve(reply["rpc"], reply)}
                except Exception as e:
                    answer = {"error": f"{type(e).__name__}: {e}"}
                proc.stdin.write(json.dumps(answer) + "\n")
                proc.stdin.flush()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def execute(self, code: str, timeout_s: float) -> tuple[bool, str]:
        previous = signal.signal(signal.SIGALRM, _raise_timeout)
        try:
            self._send({"code": code})
            reply = self._serve(timeout_s)
            return bool(reply["ok"]), str(reply["out"])
        except (TimeoutError, BrokenPipeError, ValueError, OSError) as e:
            self.close()
            return False, f"Error: kernel {e}; the namespace was reset"
        finally:
            signal.signal(signal.SIGALRM, previous)

    def bind(self, name: str, out: str, code: int) -> None:
        self._send({"bind": name, "value": {"out": out, "code": code}})

    def checkpoint(self) -> str:
        if not self.alive:
            return ""
        previous = signal.signal(signal.SIGALRM, _raise_timeout)
        try:
            self._send({"dump": True})
            return str(self._serve(LIMITS.kernel_checkpoint_timeout_s)["out"])
        except (TimeoutError, BrokenPipeError, ValueError, OSError):
            self.close()
            return ""
        finally:
            signal.signal(signal.SIGALRM, previous)

    def restore(self, blob: str) -> None:
        if blob:
            self._send({"load": blob})

    def close(self) -> None:
        if self.alive:
            os.killpg(self._proc.pid, signal.SIGKILL)
            self._proc.wait()
        self._proc = None


def _raise_timeout(signum: int, frame: Any) -> None:
    raise TimeoutError("timed out")


_MS_PER_SECOND = 1000


class Python(Tool):
    name = "python"
    description = (
        "Run Python in a persistent sandboxed kernel; variables survive between calls and only what you print reaches you. "
        "Bound: obs('§id') returns a stored result as text, sh(cmd) runs a shell command and returns Run(out, code, lines), "
        "query(dsl) runs a read-only supergraph query. Compute over big outputs here and print only what you need."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "description": {"type": "string", "description": "Why, one short line."},
            "timeout_ms": {"type": "integer", "default": LIMITS.shell_timeout_ms, "maximum": LIMITS.shell_max_timeout_ms},
        },
        "required": ["code", "description"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.SHELL, Permission.PROMPT, "Runs Python in the sandbox.")
    output_category = Category.PROCESS

    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        timeout = min(int(args.get("timeout_ms") or LIMITS.shell_timeout_ms), LIMITS.shell_max_timeout_ms) / _MS_PER_SECOND
        ok, out = self.kernel.execute(str(args["code"]), timeout)
        out = out.rstrip("\n") or "(no output)"
        return Result.success(out) if ok else Result.error(out if out.startswith("Error:") else f"Error: {out}")
