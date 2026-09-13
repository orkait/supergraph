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
from superclaw.report import context_report
from superclaw.runtime import clip
from superclaw.settings import LIMITS, PROVIDERS, Glyphs, Settings
from superclaw.skills import load_skills

SCHEMA_VERSION = 1
_NAME_WIDTH = 18
_TOKENS_WIDTH = 9


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
    sid = resolve_session(rt, args.resume)
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
    for s in rt.store.recent():
        print(f"{s['id']}  {s['event_count']:4d} events  {s['model']}  {s['cwd']}")
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

    SuperclawApp(rt, resolve_session(rt, args.resume)).run()
    return 0


def build_parser(defaults: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="superclaw", description="A terminal coding agent with supergraph as its memory.")
    parser.add_argument("-C", "--cwd", default=".", help="workspace root (default: current directory)")
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=defaults.mode)
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--db", default=str(defaults.db_path), help="supergraph store path")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--context-window", type=int, default=defaults.context_window, help="override the model's catalog context window (0 = from catalog)")
    parser.add_argument("--budget-tokens", type=int, default=defaults.budget_tokens, help="stop a run once this many tokens were spent (0 = unlimited)")
    parser.add_argument("--budget-usd", type=float, default=defaults.budget_usd, help="stop a run once this much was spent at catalog prices (0 = unlimited)")
    parser.add_argument("--intent-gate", action="store_true", help="classify each request as answer, diagnose, change or monitor and restrict tools accordingly")
    parser.add_argument("--trust-workspace", action="store_true", help="also run hooks from <workspace>/.superclaw/hooks.json")
    parser.add_argument("--resume", default=None, help="session id, or 'latest'")
    sub = parser.add_subparsers(dest="command")
    ex = sub.add_parser("exec", help="run one prompt headless and exit")
    ex.add_argument("prompt", help="the prompt, or - to read stdin")
    ex.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text")
    ex.add_argument("--require-completion", action="store_true", help="refuse a no-tool answer while plan items are pending")
    ex.add_argument("--verify", action="store_true", help="run a read-only verifier call before accepting the final answer; implies --require-completion")
    sub.add_parser("sessions", help="list sessions")
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
    if args.command is None and not sys.stdin.isatty():
        sys.exit('superclaw: the interactive shell needs a terminal (stdin is not a TTY); for non-interactive use run: superclaw exec "<prompt>"')
    settings = replace(defaults, model=args.model, mode=args.mode, context_window=args.context_window,
                       budget_tokens=args.budget_tokens, budget_usd=args.budget_usd, db_path=Path(args.db))
    if args.command == "setup":
        return cmd_setup(settings, args)
    if args.command == "models":
        return cmd_models(settings, args)
    try:
        rt = build_runtime(settings, workspace, Mode(args.mode), max_turns=args.max_turns, intent_gate=args.intent_gate,
                           hooks=build_hooks(settings, workspace, args.trust_workspace), require_provider=args.command is not None)
    except NoProviderKey as e:
        sys.exit(f"superclaw: {e}")
    handler = {"exec": cmd_exec, "sessions": cmd_sessions, "skills": cmd_skills, "context": cmd_context}.get(args.command, cmd_tui)
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
