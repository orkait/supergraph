from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from superclaw.app import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MODEL,
    NoProviderKey,
    Runtime,
    build_runtime,
    default_db_path,
    resolve_session,
    run_once,
)
from superclaw.policy import Mode
from superclaw.skills import default_roots, load_skills

SCHEMA_VERSION = 1


def _progress_line(event: dict[str, Any]) -> str | None:
    kind = event["type"]
    if kind == "tool_call":
        args = json.dumps(event["args"])
        return f"  → {event['name']} {args[:120]}{'…' if len(args) > 120 else ''}"
    if kind == "tool_result" and not event["ok"]:
        first = event["output"].splitlines()[0][:160] if event["output"] else ""
        return f"  ✗ {event['name']}: {first}"
    if kind == "compaction":
        return f"  (compacted {event['removed']} messages)"
    return None


def cmd_exec(rt: Runtime, args: argparse.Namespace) -> int:
    sid = resolve_session(rt, args.resume)
    prompt = args.prompt if args.prompt != "-" else sys.stdin.read()
    run_id = f"run_{secrets.token_hex(4)}"
    stream = args.output_format == "stream-json"

    def emit(event: dict[str, Any]) -> None:
        if stream:
            sys.stdout.write(json.dumps({"schemaVersion": SCHEMA_VERSION, "runId": run_id, **event}) + "\n")
            sys.stdout.flush()
        elif args.output_format == "text" and (line := _progress_line(event)):
            print(line, file=sys.stderr)

    emit({"type": "run_start", "sessionId": sid, "cwd": str(rt.workspace), "model": rt.model, "mode": rt.mode.value})
    res = run_once(rt, prompt, sid, on_event=emit, require_completion=args.require_completion)
    status = "incomplete" if res.incomplete else "success"
    exit_code = 2 if res.incomplete else 0
    if stream:
        emit({"type": "final", "text": res.final_answer, "incomplete": res.incomplete, "reason": res.incomplete_reason})
        emit({"type": "run_end", "status": status, "turns": res.turns, "exitCode": exit_code})
    elif args.output_format == "json":
        print(json.dumps({"sessionId": sid, "status": status, "turns": res.turns, "final": res.final_answer,
                          "incomplete": res.incomplete, "reason": res.incomplete_reason}, indent=2))
    else:
        print(res.final_answer)
    return exit_code


def cmd_sessions(rt: Runtime, args: argparse.Namespace) -> int:
    for s in rt.store.list():
        print(f"{s['id']}  {s['event_count']:4d} events  {s['model']}  {s['cwd']}")
    return 0


def cmd_skills(rt: Runtime, args: argparse.Namespace) -> int:
    for s in load_skills(default_roots(rt.workspace)):
        print(f"{s.name}: {s.description}")
    return 0


def cmd_tui(rt: Runtime, args: argparse.Namespace) -> int:
    from superclaw.tui import SuperclawApp

    SuperclawApp(rt, resolve_session(rt, args.resume)).run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="superclaw", description="A terminal coding agent with supergraph as its memory.")
    parser.add_argument("-C", "--cwd", default=".", help="workspace root (default: current directory)")
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=os.environ.get("SUPERCLAW_MODE", Mode.ASK.value))
    parser.add_argument("--model", default=os.environ.get("SUPERCLAW_MODEL", DEFAULT_MODEL))
    parser.add_argument("--db", default=None, help="supergraph store path (default: $SUPERCLAW_DB_PATH or ~/.local/share/superclaw/brain)")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--context-window", type=int, default=int(os.environ.get("SUPERCLAW_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW)))
    parser.add_argument("--resume", default=None, help="session id, or 'latest'")
    sub = parser.add_subparsers(dest="command")
    ex = sub.add_parser("exec", help="run one prompt headless and exit")
    ex.add_argument("prompt", help="the prompt, or - to read stdin")
    ex.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text")
    ex.add_argument("--require-completion", action="store_true", help="refuse a no-tool answer while plan items are pending")
    sub.add_parser("sessions", help="list sessions")
    sub.add_parser("skills", help="list discovered skills")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.cwd).resolve()
    if not workspace.is_dir():
        sys.exit(f"superclaw: not a directory: {workspace}")
    try:
        rt = build_runtime(workspace, Mode(args.mode), args.model, Path(args.db) if args.db else default_db_path(),
                           max_turns=args.max_turns, context_window=args.context_window)
    except NoProviderKey as e:
        sys.exit(f"superclaw: {e}")
    handler = {"exec": cmd_exec, "sessions": cmd_sessions, "skills": cmd_skills}.get(args.command, cmd_tui)
    try:
        return handler(rt, args)
    except KeyError as e:
        sys.exit(f"superclaw: {e.args[0]}")
    finally:
        rt.close()


if __name__ == "__main__":
    sys.exit(main())
