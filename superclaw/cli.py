from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from superclaw.app import Callbacks, NoProviderKey, Runtime, build_hooks, build_runtime, resolve_session, run_once
from superclaw.catalog import describe, keyed_providers, models_for
from superclaw.policy import Mode
from superclaw.provider import hint
from superclaw.report import context_report, doctor_lines
from superclaw.runtime import clip
from superclaw.settings import LIMITS, PROVIDERS, Glyphs, Settings
from superclaw.skills import load_skills
from superclaw.worktree import WorktreeError
from superclaw.worktree import prepare as prepare_worktree

SCHEMA_VERSION = 1
_NAME_WIDTH = 18
_TOKENS_WIDTH = 9
_EVENT_TYPE_WIDTH = 12


def _tool_set(value: str) -> frozenset[str]:
    return frozenset(t for t in value.replace(",", " ").split() if t)


def _progress_line(event: dict[str, Any], glyphs: Glyphs) -> str | None:
    kind = event["type"]
    if kind == "tool_call":
        args = json.dumps(event["args"])
        return f"  {glyphs.call} {event['name']} {clip(args, LIMITS.preview_args_chars)}"
    if kind == "tool_result" and not event["ok"]:
        first = event["output"].splitlines()[0] if event["output"] else ""
        return f"  {glyphs.failed} {event['name']}: {clip(first, LIMITS.preview_error_chars)}"
    if kind == "compaction":
        return f"  (compacted {event['removed']} messages)"
    return None


def cmd_exec(rt: Runtime, args: argparse.Namespace) -> int:
    sid = resolve_session(rt, args.resume, args.fork)
    prompt = args.prompt if args.prompt != "-" else sys.stdin.read()
    run_id = f"run_{secrets.token_hex(LIMITS.run_id_bytes)}"
    stream = args.output_format == "stream-json"

    def emit(event: dict[str, Any]) -> None:
        if stream:
            sys.stdout.write(json.dumps({"schemaVersion": SCHEMA_VERSION, "runId": run_id, **event}) + "\n")
            sys.stdout.flush()
        elif args.output_format == "text" and (line := _progress_line(event, rt.settings.glyphs)):
            print(line, file=sys.stderr)

    emit({"type": "run_start", "sessionId": sid, "cwd": str(rt.workspace), "model": rt.model, "mode": rt.mode.value})
    res = run_once(rt, prompt, sid, Callbacks(on_event=emit), require_completion=args.require_completion or args.verify, verify=args.verify)
    status = "incomplete" if res.incomplete else "success"
    exit_code = 2 if res.incomplete else 0
    if stream:
        emit({"type": "final", "text": res.final_answer, "incomplete": res.incomplete, "reason": res.incomplete_reason})
        emit({"type": "run_end", "status": status, "turns": res.turns, "exitCode": exit_code,
              "savedTokens": res.saved_tokens, "keptOutTokens": res.kept_out_tokens})
    elif args.output_format == "json":
        print(json.dumps({"sessionId": sid, "status": status, "turns": res.turns, "final": res.final_answer,
                          "incomplete": res.incomplete, "reason": res.incomplete_reason}, indent=2))
    else:
        print(res.final_answer)
    return exit_code


def cmd_sessions(rt: Runtime, args: argparse.Namespace) -> int:
    if args.query:
        hits = rt.store.search(args.query)
        for hit in hits:
            print(f"{hit['id']}  #{hit['seq']:<4d} {hit['type']:<{_EVENT_TYPE_WIDTH}} {hit['text']}")
        return 0 if hits else 1
    for s in rt.store.recent():
        print(f"{s['id']}  {s['event_count']:4d} events  {s['model']}  {s['cwd']}")
    return 0


def cmd_doctor(rt: Runtime, args: argparse.Namespace) -> int:
    for line in doctor_lines(rt, "run `superclaw setup`"):
        print(line)
    return 0


def cmd_usage(rt: Runtime, args: argparse.Namespace) -> int:
    for s in rt.store.recent():
        u = rt.store.usage(s["id"])
        print(f"{s['id']}  {u['calls']:4d} calls  {u['tokens']:>10,} tokens  ${u['cost_usd']:.4f}  {s['model']}")
    return 0


def cmd_skills(rt: Runtime, args: argparse.Namespace) -> int:
    for s in load_skills(rt.settings.skill_roots(rt.workspace)):
        print(f"{s.name}: {s.description}")
    return 0


def cmd_context(rt: Runtime, args: argparse.Namespace) -> int:
    report = context_report(rt, args.prompt)
    info = rt.model_info
    print(f"{rt.model}  window {report.window:,}  max output {info.max_output_tokens:,}  {'catalog' if info.known else 'fallback (unknown model)'}")
    for name, tokens in [*report.categories.items(), ("free", report.free)]:
        print(f"  {name:<{_NAME_WIDTH}}{tokens:>{_TOKENS_WIDTH},}  {report.percent(tokens):5.1f}%")
    return 0


def cmd_tui(rt: Runtime, args: argparse.Namespace) -> int:
    from superclaw.tui import SuperclawApp

    SuperclawApp(rt, resolve_session(rt, args.resume, args.fork)).run()
    return 0


def build_parser(defaults: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="superclaw", description="A terminal coding agent with supergraph as its memory.")
    parser.add_argument("-C", "--cwd", default=".", help="workspace root (default: current directory)")
    parser.add_argument("--add-dir", action="append", default=[], metavar="PATH",
                        help="allow reads and writes in an extra directory, and bind it into the sandbox (repeatable)")
    parser.add_argument("-w", "--worktree", nargs="?", const="", default=None, metavar="NAME",
                        help="run in an isolated git worktree on branch superclaw/<name> (default name: task-<utc timestamp>)")
    parser.add_argument("--worktree-dir", default="", metavar="PATH", help="base directory for created worktrees")
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=defaults.mode)
    parser.add_argument("--dangerously-skip-permissions", action="store_true",
                        help="run every tool without asking (same as --mode unsafe); only inside a sandbox you can discard")
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--db", default=str(defaults.db_path), help="supergraph store path")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--context-window", type=int, default=defaults.context_window, help="override the model's catalog context window (0 = from catalog)")
    parser.add_argument("--budget-tokens", type=int, default=defaults.budget_tokens, help="stop a run once this many tokens were spent (0 = unlimited)")
    parser.add_argument("--budget-usd", type=float, default=defaults.budget_usd, help="stop a run once this much was spent at catalog prices (0 = unlimited)")
    parser.add_argument("--intent-gate", action="store_true", help="classify each request as answer, diagnose, change or monitor and restrict tools accordingly")
    parser.add_argument("--trust-workspace", action="store_true", help="also run hooks from <workspace>/.superclaw/hooks.json")
    parser.add_argument("--allow-tools", default="", help="only expose these tools (comma or space separated)")
    parser.add_argument("--deny-tools", default="", help="hide these tools (comma or space separated)")
    parser.add_argument("--resume", default=None, help="session id, or 'latest'")
    parser.add_argument("--fork", default=None, help="copy a session (id or 'latest') into a new one and continue from it")
    sub = parser.add_subparsers(dest="command")
    ex = sub.add_parser("exec", help="run one prompt headless and exit")
    ex.add_argument("prompt", help="the prompt, or - to read stdin")
    ex.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text")
    ex.add_argument("--require-completion", action="store_true", help="refuse a no-tool answer while plan items are pending")
    ex.add_argument("--verify", action="store_true", help="run a read-only verifier call before accepting the final answer; implies --require-completion")
    sess = sub.add_parser("sessions", help="list sessions, or search their events")
    sess.add_argument("query", nargs="?", default="", help="search text; omit to list recent sessions")
    sub.add_parser("doctor", help="terminal, sandbox, model, store and provider health")
    sub.add_parser("usage", help="token and cost totals per recent session")
    sub.add_parser("skills", help="list discovered skills")
    ctx = sub.add_parser("context", help="show what the first request would cost in context tokens")
    ctx.add_argument("prompt", nargs="?", default="", help="optional prompt, used for memory recall")
    setup = sub.add_parser("setup", help="store a provider key and default model")
    setup.add_argument("--provider", choices=[p.name for p in PROVIDERS], default=PROVIDERS[0].name)
    setup.add_argument("--key", default="", help="the API key; prompted when omitted")
    models = sub.add_parser("models", help="list the models each connected provider serves")
    models.add_argument("--provider", choices=[p.name for p in PROVIDERS], default="", help="one provider instead of every one with a key")
    models.add_argument("--refresh", action="store_true", help="ignore the cached listing and ask the provider again")
    return parser


def cmd_models(settings: Settings, args: argparse.Namespace) -> int:
    providers = [p for p in PROVIDERS if p.name == args.provider] if args.provider else keyed_providers()
    if not providers:
        sys.exit("superclaw: no provider key found; run `superclaw setup`")
    for provider in providers:
        for model in models_for(provider, os.environ.get(provider.env, ""), settings.models_cache, refresh=args.refresh):
            mark = "*" if model.id == settings.model else " "
            print(f"{mark} {model.id:<{LIMITS.model_id_width}} {describe(model, ' ')}")
    return 0


def cmd_setup(settings: Settings, args: argparse.Namespace) -> int:
    provider = next(p for p in PROVIDERS if p.name == args.provider)
    key = args.key or getpass.getpass(f"{provider.name} API key ({provider.console}): ")
    if not key.strip():
        sys.exit("superclaw: no key entered")
    model = settings.model if settings.model.startswith(provider.name + "/") or provider.name == PROVIDERS[0].name else provider.default_model
    settings.save_credentials(provider, key.strip(), model)
    print(f"saved {provider.env} and SUPERCLAW_MODEL={model} to {settings.credentials}")
    return 0


def main(argv: list[str] | None = None) -> int:
    defaults = Settings.from_env()
    args = build_parser(defaults).parse_args(argv)
    workspace = Path(args.cwd).resolve()
    if not workspace.is_dir():
        sys.exit(f"superclaw: not a directory: {workspace}")
    extra_dirs = tuple(Path(d).resolve() for d in args.add_dir)
    for extra in extra_dirs:
        if not extra.is_dir():
            sys.exit(f"superclaw: not a directory: {extra}")
    if args.command is None and not sys.stdin.isatty():
        sys.exit('superclaw: the interactive shell needs a terminal (stdin is not a TTY); for non-interactive use run: superclaw exec "<prompt>"')
    mode = Mode.UNSAFE.value if args.dangerously_skip_permissions else args.mode
    settings = replace(defaults, model=args.model, mode=mode, context_window=args.context_window,
                       budget_tokens=args.budget_tokens, budget_usd=args.budget_usd, db_path=Path(args.db))
    if args.command == "setup":
        return cmd_setup(settings, args)
    if args.command == "models":
        return cmd_models(settings, args)
    if args.worktree is not None:
        try:
            tree = prepare_worktree(workspace, Path(args.worktree_dir) if args.worktree_dir else settings.worktrees_dir, args.worktree)
        except WorktreeError as e:
            sys.exit(f"superclaw: {e}")
        print(f"superclaw: {'reusing' if tree.reused else 'created'} worktree {tree.path} on {tree.branch}", file=sys.stderr)
        workspace = tree.path
    try:
        rt = build_runtime(settings, workspace, Mode(mode), max_turns=args.max_turns, intent_gate=args.intent_gate,
                           hooks=build_hooks(settings, workspace, args.trust_workspace), require_provider=args.command not in (None, "doctor"),
                           allow_tools=_tool_set(args.allow_tools), deny_tools=_tool_set(args.deny_tools), extra_dirs=extra_dirs)
    except NoProviderKey as e:
        sys.exit(f"superclaw: {e}")
    handler = {"exec": cmd_exec, "sessions": cmd_sessions, "usage": cmd_usage, "skills": cmd_skills,
               "context": cmd_context, "doctor": cmd_doctor}.get(args.command, cmd_tui)
    try:
        return handler(rt, args)
    except KeyError as e:
        sys.exit(f"superclaw: {e.args[0]}")
    except RuntimeError as e:
        advice = hint(str(e), tui=False)
        sys.exit(f"superclaw: {e}" + (f"\n  {advice}" if advice else ""))
    finally:
        rt.close()


if __name__ == "__main__":
    sys.exit(main())
