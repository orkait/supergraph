from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from superclaw import checks, cron, maintain, plugins, repomap, review, spec, update
from superclaw.acp import serve as acp_serve
from superclaw.agents import load_agents
from superclaw.agents import resolve as resolve_agent
from superclaw.app import Callbacks, NoProviderKey, Runtime, build_hooks, build_runtime, mcp_paths, resolve_session, run_once, switch_model
from superclaw.attach import read as read_attachments
from superclaw.catalog import describe, keyed_providers, models_for
from superclaw.facts import as_of_ms
from superclaw.loop import Result
from superclaw.mcp import MCPError, add_server, remove_server
from superclaw.policy import Mode
from superclaw.prompt import _git_branch
from superclaw.provider import hint
from superclaw.report import context_report, doctor_lines
from superclaw.schema import SchemaError
from superclaw.schema import extract as schema_extract
from superclaw.schema import instruction as schema_instruction
from superclaw.schema import load as load_schema
from superclaw.schema import problems as schema_problems
from superclaw.serve import serve as serve_mcp
from superclaw.settings import (
    LIMITS,
    MCP_FILE,
    MCP_SCOPES,
    PROVIDERS,
    RENDERER_INLINE,
    RENDERERS,
    SERVED_TOOLS,
    SESSION_END_EXIT,
    SESSION_END_OTHER,
    WORKSPACE_DIR,
    Glyphs,
    Settings,
    split_models,
)
from superclaw.share import NotServing
from superclaw.skills import load_skills
from superclaw.text import clip
from superclaw.tools import jail
from superclaw.usercommands import expand, load_commands
from superclaw.usercommands import find as find_command
from superclaw.worktree import WorktreeError
from superclaw.worktree import prepare as prepare_worktree
from supergraph.core.errors import StoreInUse, SuperGraphError

SCHEMA_VERSION = 1
STORELESS = ("doctor", "mcp", "agents", "skills", "commands")
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
    if prompt.startswith("/"):
        name, _, rest = prompt[1:].partition(" ")
        command = find_command(name, rt.settings.command_roots(rt.workspace), rt.settings.skill_roots(rt.workspace))
        if command is None:
            print(f"superclaw: no user command or skill /{name}; `superclaw commands` lists them, sending the text as typed", file=sys.stderr)
        else:
            if command.agent:
                rt.agent = resolve_agent(command.agent, rt.settings.agent_roots(rt.workspace))
                rt.policy.scope_to(rt.agent.tools)
            if command.model:
                switch_model(rt, command.model)
            prompt = expand(command.template, rest)
    attached = read_attachments(args.file, (rt.workspace, *rt.extra_dirs))
    for problem in attached.problems:
        print(f"superclaw: attachment {problem}", file=sys.stderr)
    if attached.text:
        prompt = f"{prompt}\n\n{attached.text}"
    shape = None
    if args.output_schema:
        try:
            shape = load_schema(Path(args.output_schema))
        except SchemaError as e:
            sys.exit(f"superclaw: {e}")
        prompt = f"{prompt}\n\n{schema_instruction(shape)}"
    run_id = f"run_{secrets.token_hex(LIMITS.run_id_bytes)}"
    stream = args.output_format == "stream-json"

    def emit(event: dict[str, Any]) -> None:
        if stream:
            sys.stdout.write(json.dumps({"schemaVersion": SCHEMA_VERSION, "runId": run_id, **event}) + "\n")
            sys.stdout.flush()
        elif args.output_format == "text" and (line := _progress_line(event, rt.settings.glyphs)):
            print(line, file=sys.stderr)

    emit({"type": "run_start", "sessionId": sid, "cwd": str(rt.workspace), "model": rt.model, "mode": rt.mode.value})
    if args.spec:
        res = draft_spec(rt, prompt, sid, emit)
    else:
        res = run_once(rt, prompt, sid, Callbacks(on_event=emit), require_completion=args.require_completion or args.verify,
                       verify=args.verify, images=attached.images)
    status = "incomplete" if res.incomplete else "success"
    exit_code = 2 if res.incomplete else 0
    if args.spec:
        status = "spec_review" if res.stop_reason == spec.CONTROL else "no_spec"
        exit_code = 3 if res.stop_reason == spec.CONTROL else 2
    if shape is not None and not res.incomplete:
        try:
            found = schema_problems(schema_extract(res.final_answer), shape)
        except SchemaError as e:
            found = [str(e)]
        for problem in found:
            print(f"superclaw: output schema: {problem}", file=sys.stderr)
        if found:
            status, exit_code = "schema_mismatch", 2
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


def cmd_cron(rt: Runtime, args: argparse.Namespace) -> int:
    store = cron.CronStore(rt.gs)
    try:
        if args.cron_command == "add":
            job = store.add(args.id, args.schedule, args.prompt if args.prompt != "-" else sys.stdin.read(), args.model)
            print(f"added {job.id}: {job.expr!r}, next run {_when(job.next_run_ms)}")
            return cron.fire(rt, store, job) if args.run_now else 0
        if args.cron_command in ("pause", "resume"):
            job = store.set_status(args.id, "paused" if args.cron_command == "pause" else "active")
            print(f"{job.id} is {job.status}; next run {_when(job.next_run_ms)}")
            return 0
        if args.cron_command == "rm":
            store.remove(args.id)
            print(f"removed {args.id}")
            return 0
        if args.cron_command == "run":
            def emit(event: dict[str, Any]) -> None:
                if event["type"].startswith("cron_") or event["type"] == "error":
                    print(f"superclaw: {event['type']} {event.get('job', '')} {event.get('reason') or event.get('message') or event.get('expr', '')}".rstrip(), file=sys.stderr)
            fired = cron.run(rt, store, tuple(args.ids), once=args.once, catch_up=args.catch_up, emit=emit)
            print(f"superclaw: fired {fired} job(s)", file=sys.stderr)
            return 0
    except cron.CronError as e:
        sys.exit(f"superclaw: {e}")
    jobs = store.list()
    for job in jobs:
        print(f"{job.id:<{_NAME_WIDTH}} {job.status:<8} {job.expr:<16} next {_when(job.next_run_ms)}  fired {job.fire_count}  {clip(job.prompt, LIMITS.preview_args_chars)}")
    if not jobs:
        print("no cron jobs; add one with `superclaw cron add <id> <cron-expr> --prompt \"...\"`", file=sys.stderr)
        return 1
    return 0


def _when(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ms else "never"


def draft_spec(rt: Runtime, prompt: str, sid: str, emit: Callable[[dict[str, Any]], None]) -> Result:
    rt.registry.register(spec.SubmitSpec())
    rt.mode = Mode.PLAN
    rt.policy.plan_exempt = frozenset({spec.TOOL_NAME})
    return run_once(rt, f"{spec.draft_prompt()}\n\n<task>\n{prompt}\n</task>", sid, Callbacks(on_event=emit), authored=False)


def cmd_spec(rt: Runtime, args: argparse.Namespace) -> int:
    if args.spec_command == "approve":
        try:
            body, path = spec.load(rt.workspace, args.id)
        except spec.SpecError as e:
            sys.exit(f"superclaw: {e}")
        sid = rt.store.create(cwd=str(rt.workspace), model=rt.model, title=f"implement {path.stem}", branch=_git_branch(rt.workspace))
        print(f"superclaw: implementing {path.name} in mode {rt.mode.value}, session {sid}", file=sys.stderr)
        res = run_once(rt, spec.implementation_prompt(body, path, args.note), sid, Callbacks(on_event=lambda event: None), authored=False)
        print(res.final_answer)
        return 2 if res.incomplete else 0
    if args.spec_command == "show":
        try:
            body, path = spec.load(rt.workspace, args.id)
        except spec.SpecError as e:
            sys.exit(f"superclaw: {e}")
        print(body)
        return 0
    found = spec.list_specs(rt.workspace)
    for path in found:
        print(f"{path.stem:<{LIMITS.model_id_width}} {path.read_text(errors='replace').splitlines()[0].lstrip('# ') if path.stat().st_size else ''}")
    if not found:
        print(f"no specs under {spec.specs_dir(rt.workspace)}; draft one with `superclaw exec --spec \"<task>\"`", file=sys.stderr)
        return 1
    return 0


def cmd_verify(rt: Runtime, args: argparse.Namespace) -> int:
    found = checks.detect(rt.workspace)
    only = tuple(_tool_set(args.only))
    if not found or (only and not any(c.id in only for c in found)):
        print("superclaw: no verification checks detected" + (f" matching {args.only}" if only else "") + "; supported: go.mod, package.json scripts, pytest, Cargo.toml", file=sys.stderr)
        return 1
    attempts = max(1, min(args.attempts, LIMITS.verify_max_attempts))
    report = checks.run(rt.workspace, found, only, args.timeout_s or LIMITS.verify_timeout_s)
    used = 1
    while not report.ok and used < attempts:
        print(f"superclaw: attempt {used} failed {len(report.failed)} check(s); asking the agent to fix it", file=sys.stderr)
        sid = rt.store.create(cwd=str(rt.workspace), model=rt.model, title=f"verify attempt {used}", branch=_git_branch(rt.workspace))
        run_once(rt, checks.remediation_prompt(report), sid, Callbacks(on_event=lambda event: None), authored=False)
        report = checks.run(rt.workspace, found, only, args.timeout_s or LIMITS.verify_timeout_s)
        used += 1
    if args.json:
        print(json.dumps({**checks.as_json(report), "attempts": used}, indent=2))
    else:
        for line in checks.lines(report):
            print(line)
    return 0 if report.ok else 1


def cmd_acp(rt: Runtime, args: argparse.Namespace) -> int:
    print(f"superclaw: serving the Agent Client Protocol over stdio for {rt.workspace}", file=sys.stderr)
    acp_serve(rt)
    return 0


def cmd_review(rt: Runtime, args: argparse.Namespace) -> int:
    try:
        if args.commit:
            change = review.commit(rt.workspace, args.commit)
        elif args.base:
            change = review.against(rt.workspace, args.base)
        else:
            change = review.uncommitted(rt.workspace)
        text = review.prompt(change, args.prompt if args.prompt != "-" else sys.stdin.read())
    except review.ReviewError as e:
        sys.exit(f"superclaw: {e}")
    rt.mode = Mode.PLAN
    sid = resolve_session(rt, None, None)
    rt.store.rename(sid, f"review: {change.label}")
    print(f"superclaw: reviewing {change.label} in plan mode, session {sid}", file=sys.stderr)
    res = run_once(rt, text, sid, Callbacks(on_event=lambda event: None), authored=False)
    print(res.final_answer)
    return 2 if res.incomplete else 0


def cmd_export(rt: Runtime, args: argparse.Namespace) -> int:
    sid = rt.store.latest() if args.session in ("", "latest") else args.session
    if not sid or rt.store.get(sid) is None:
        sys.exit(f"superclaw: no session {args.session or 'latest'!r}")
    doc = json.dumps(rt.store.export(sid), indent=2)
    if args.output:
        Path(args.output).write_text(doc + "\n")
        print(f"superclaw: wrote session {sid} to {args.output}", file=sys.stderr)
    else:
        print(doc)
    return 0


def cmd_import(rt: Runtime, args: argparse.Namespace) -> int:
    try:
        doc = json.loads(Path(args.file).read_text() if args.file != "-" else sys.stdin.read())
        sid = rt.store.import_(doc, cwd=str(rt.workspace))
    except (OSError, ValueError) as e:
        sys.exit(f"superclaw: {e}")
    print(sid)
    return 0


def cmd_sessions(rt: Runtime, args: argparse.Namespace) -> int:
    if args.touching:
        path = str(jail(rt.workspace, args.touching))
        found = rt.store.touching(path)
        for s in found:
            verbs = sorted({verb for file, verb in rt.store.files_of(s["id"]) if file == path})
            print(f"{s['id']}  {', '.join(verbs):<12} {s.get('title') or ''}")
        return 0 if found else 1
    if args.query:
        hits = rt.store.search(args.query)
        for hit in hits:
            print(f"{hit['id']}  #{hit['seq']:<4d} {hit['type']:<{_EVENT_TYPE_WIDTH}} {hit['text']}")
        return 0 if hits else 1
    for s in rt.store.recent():
        print(f"{s['id']}  {s['event_count']:4d} events  {s['model']}  {s['cwd']}")
    return 0


def cmd_facts(rt: Runtime, args: argparse.Namespace) -> int:
    facts = rt.memory.facts
    if args.retract:
        facts.retract(args.retract, args.reason or "retracted by the user")
        print(f"retracted {args.retract}")
        return 0
    as_of = as_of_ms(args.as_of) if args.as_of else None
    if args.as_of and as_of is None:
        print("--as-of takes an ISO date such as 2026-09-01", file=sys.stderr)
        return 2
    found = facts.search(args.query, as_of=as_of) if args.query else facts.recent(as_of=as_of)
    for fact in found:
        print(f"{fact.id}  {fact.line()}")
    return 0 if found else 1


def cmd_maintain(rt: Runtime, args: argparse.Namespace) -> int:
    report = maintain.maintain(rt.gs, optimize=not args.no_optimize)
    print(report.line(rt.settings.glyphs.dot))
    print(maintain.health_line(rt.gs, rt.settings.glyphs.dot))
    return 0


def cmd_ask(rt: Runtime, args: argparse.Namespace) -> int:
    try:
        answer = rt.memory.facts.ask(args.question)
    except SuperGraphError as e:
        print(f"superclaw: {e}", file=sys.stderr)
        return 1
    print(answer.text)
    if answer.cited:
        print("cited: " + ", ".join(answer.cited))
    return 0


def cmd_mcp(rt: Runtime, args: argparse.Namespace) -> int:
    if args.mcp_command == "serve":
        print(f"superclaw: serving {', '.join(SERVED_TOOLS)} from {rt.settings.db_path} ({rt.gs.role}) over MCP stdio", file=sys.stderr)
        serve_mcp(rt)
        return 0
    bridge = rt.mcp
    for tool in bridge.tools if bridge else []:
        print(f"{tool.server:<{_NAME_WIDTH}} {tool.name:<{LIMITS.model_id_width}} {tool.summary()}")
    for skipped in bridge.skipped if bridge else []:
        print(f"{skipped.name:<{_NAME_WIDTH}} skipped: {skipped.error}", file=sys.stderr)
    for problem in bridge.problems if bridge else []:
        print(f"config: {problem}", file=sys.stderr)
    if bridge is None or not bridge.tools:
        print(f"no MCP tools; add servers to {rt.settings.user_mcp}", file=sys.stderr)
        return 1
    return 0


def cmd_doctor(rt: Runtime, args: argparse.Namespace) -> int:
    for line in doctor_lines(rt, "run `superclaw setup`"):
        print(line)
    return 0


def cmd_usage(rt: Runtime, args: argparse.Namespace) -> int:
    for s in rt.store.recent():
        u = rt.store.usage(s["id"])
        print(f"{s['id']}  {u['calls']:4d} calls  {u['tokens']:>10,} tokens  {u['cached']:>10,} cached  ${u['cost_usd']:.4f}  {s['model']}")
    return 0


def cmd_agents(rt: Runtime, args: argparse.Namespace) -> int:
    found = load_agents(rt.settings.agent_roots(rt.workspace))
    for agent in found:
        tools = " ".join(sorted(agent.tools)) or "all tools"
        print(f"{agent.name:<{_NAME_WIDTH}} {agent.description}")
        print(f"{'':<{_NAME_WIDTH}} {agent.model or 'default model'}  {tools}  ({agent.path})")
    if not found:
        print(f"no agent profiles; add <name>.md to {rt.settings.agent_roots()[0]}", file=sys.stderr)
        return 1
    return 0


def cmd_commands(rt: Runtime, args: argparse.Namespace) -> int:
    found = load_commands(rt.settings.command_roots(rt.workspace), rt.settings.skill_roots(rt.workspace))
    for command in found:
        routing = " ".join(part for part in ("skill" if command.skill else "", f"agent={command.agent}" if command.agent else "", f"model={command.model}" if command.model else "") if part)
        print(f"/{command.name:<{_NAME_WIDTH}} {command.description}  {routing}({command.path})")
    if not found:
        print(f"no user commands; add <name>.md to {rt.settings.command_roots()[0]}", file=sys.stderr)
        return 1
    return 0


def cmd_skills(rt: Runtime, args: argparse.Namespace) -> int:
    for s in load_skills(rt.settings.skill_roots(rt.workspace)):
        print(f"{s.name}: {s.description}")
    return 0


def cmd_context(rt: Runtime, args: argparse.Namespace) -> int:
    report = context_report(rt, args.prompt)
    info = rt.model_info
    print(f"{rt.model}  window {report.window:,}  max output {info.max_output_tokens:,}  cap {rt.settings.output_cap():,}  {'catalog' if info.known else 'fallback (unknown model)'}")
    for name, tokens in [*report.categories.items(), ("free", report.free)]:
        print(f"  {name:<{_NAME_WIDTH}}{tokens:>{_TOKENS_WIDTH},}  {report.percent(tokens):5.1f}%")
    return 0


def cmd_tui(rt: Runtime, args: argparse.Namespace) -> int:
    from superclaw.tui import SuperclawApp

    renderer = args.tui or rt.settings.renderer
    app = SuperclawApp(rt, resolve_session(rt, args.resume, args.fork))
    app.run(inline=renderer == RENDERER_INLINE, inline_no_clear=True)
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
    parser.add_argument("--tui", choices=RENDERERS, default=None,
                        help=f"renderer for the interactive shell: default draws under your prompt and keeps scrollback, "
                             f"fullscreen takes the alternate screen (saved default: {defaults.renderer})")
    parser.add_argument("--model", default="", help=f"model for this session (default: {defaults.model})")
    parser.add_argument("--agent", default="", metavar="NAME",
                        help="agent profile from <config>/agents or <workspace>/.superclaw/agents; `agents` lists them")
    parser.add_argument("--fallback-model", default="", metavar="MODELS",
                        help="comma or space separated models to try, in order, when the main model fails")
    parser.add_argument("--db", default=str(defaults.db_path), help="supergraph store path")
    parser.add_argument("--max-turns", type=int, default=LIMITS.max_turns, metavar="N", help="stop after N model turns and ask for a final answer; 0 (default) is no cap, the loop guards and budgets end a run instead")
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
    ex.add_argument("-f", "--file", action="append", default=[], metavar="PATH",
                    help="attach a workspace file to the prompt; an image is sent as an image (repeatable)")
    ex.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text")
    ex.add_argument("--output-schema", default="", metavar="FILE",
                    help="JSON Schema the final answer must match; a mismatch exits 2")
    ex.add_argument("--spec", action="store_true",
                    help="draft an implementation spec read-only, save it under .superclaw/specs and stop for review (exit 3); approve with `superclaw spec approve`")
    cr = sub.add_parser("cron", help="schedule prompts: add, list, pause, resume, rm, and run the scheduler in the foreground")
    cr_sub = cr.add_subparsers(dest="cron_command")
    cr_add = cr_sub.add_parser("add", help="add a job")
    cr_add.add_argument("id")
    cr_add.add_argument("schedule", help='five-field cron expression, or @hourly, @daily, @weekly, @monthly')
    cr_add.add_argument("--prompt", required=True, help="the prompt to run, or - for stdin")
    cr_add.add_argument("--model", default="", help="model for this job (default: the session model)")
    cr_add.add_argument("--run-now", action="store_true", help="fire it once immediately after adding")
    cr_sub.add_parser("list", help="jobs with status, schedule and next run")
    cr_sub.add_parser("pause", help="stop a job from firing").add_argument("id")
    cr_sub.add_parser("resume", help="let a paused job fire again").add_argument("id")
    cr_sub.add_parser("rm", help="delete a job").add_argument("id")
    cr_run = cr_sub.add_parser("run", help="run due jobs; without --once, keep running in the foreground")
    cr_run.add_argument("ids", nargs="*", help="only these jobs")
    cr_run.add_argument("--once", action="store_true", help="fire every currently due job once and exit")
    cr_run.add_argument("--catch-up", action="store_true", help="fire jobs that became due while no scheduler was running, instead of skipping to their next slot")
    sp = sub.add_parser("spec", help="list, show or approve saved implementation specs")
    sp_sub = sp.add_subparsers(dest="spec_command")
    sp_sub.add_parser("list", help="specs under .superclaw/specs")
    sp_sub.add_parser("show", help="print a spec").add_argument("id")
    approve = sp_sub.add_parser("approve", help="implement a spec in the current mode")
    approve.add_argument("id")
    approve.add_argument("--note", default="", help="a note for the implementer, appended to the spec")
    ex.add_argument("--require-completion", action="store_true", help="refuse a no-tool answer while plan items are pending")
    ex.add_argument("--verify", action="store_true", help="run a read-only verifier call before accepting the final answer; implies --require-completion")
    vf = sub.add_parser("verify", help="detect and run the workspace's checks (go test, package.json scripts, pytest, cargo test)")
    vf.add_argument("--only", default="", metavar="IDS", help="run only these check ids, comma or space separated")
    vf.add_argument("--timeout-s", type=int, default=0, help=f"per-check timeout (default {LIMITS.verify_timeout_s})")
    vf.add_argument("--attempts", type=int, default=LIMITS.verify_attempts,
                    help=f"after a failure, let the agent fix it and rerun, up to this many attempts (max {LIMITS.verify_max_attempts})")
    vf.add_argument("--json", action="store_true")
    sub.add_parser("acp", help="serve the Agent Client Protocol over stdio so an editor can drive superclaw")
    rv = sub.add_parser("review", help="review a change read-only and print findings with file:line and a verdict")
    rv.add_argument("prompt", nargs="?", default="", help="extra focus for the reviewer, or - to read it from stdin")
    scope = rv.add_mutually_exclusive_group()
    scope.add_argument("--uncommitted", action="store_true", help="staged, unstaged and untracked changes (default)")
    scope.add_argument("--base", default="", metavar="BRANCH", help="changes on this branch since it left BRANCH")
    scope.add_argument("--commit", default="", metavar="SHA", help="the changes one commit introduced")
    exp = sub.add_parser("export", help="write a session and its events as JSON")
    exp.add_argument("session", nargs="?", default="latest", help="session id (default: latest)")
    exp.add_argument("-o", "--output", default="", metavar="FILE", help="write here instead of stdout")
    imp = sub.add_parser("import", help="create a new session from an exported JSON file")
    imp.add_argument("file", help="the export, or - for stdin")
    maint = sub.add_parser("maintain", help="expire stale nodes, decay old facts, optimize the store and print its health")
    maint.add_argument("--no-optimize", action="store_true", help="skip SYS OPTIMIZE")
    ask = sub.add_parser("ask", help="answer a question from stored facts and results with the substrate's reader, no agent loop")
    ask.add_argument("question")
    facts = sub.add_parser("facts", help="facts learned from sources, with age and URL; search, view as of a date, or retract one")
    facts.add_argument("query", nargs="?", default="", help="search text; omit to list the newest")
    facts.add_argument("--as-of", default="", help="only facts observed on or before this ISO date")
    facts.add_argument("--retract", default="", help="fact id to retract")
    facts.add_argument("--reason", default="", help="why it is retracted")
    sess = sub.add_parser("sessions", help="list sessions, or search their events")
    sess.add_argument("query", nargs="?", default="", help="search text; omit to list recent sessions")
    sess.add_argument("--touching", default="", help="list the sessions that read or wrote this workspace path")
    sub.add_parser("doctor", help="terminal, sandbox, model, store and provider health")
    mcp = sub.add_parser("mcp", help="list the configured MCP servers and the tools they expose, or add and remove servers")
    mcp_sub = mcp.add_subparsers(dest="mcp_command")
    mcp_sub.add_parser("list", help="connect to every configured server and list its tools")
    mcp_sub.add_parser("serve", help=f"serve this brain to another agent over MCP stdio: {', '.join(SERVED_TOOLS)}")
    mcp_add = mcp_sub.add_parser("add", help="add a server: `mcp add NAME -- CMD ARGS...` for stdio, `mcp add NAME --url URL` for streamable HTTP")
    mcp_add.add_argument("name")
    mcp_add.add_argument("argv", nargs="*", metavar="CMD", help="the stdio command and its arguments, after --")
    mcp_add.add_argument("--url", default="", help="streamable HTTP endpoint instead of a command")
    mcp_add.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="environment for a stdio server (repeatable)")
    mcp_add.add_argument("--header", action="append", default=[], metavar="KEY=VALUE", help="HTTP header for a url server (repeatable)")
    mcp_add.add_argument("--scope", choices=MCP_SCOPES, default=MCP_SCOPES[0], help=f"user writes {MCP_FILE} in the config dir, project writes <workspace>/{WORKSPACE_DIR}/{MCP_FILE}")
    mcp_remove = mcp_sub.add_parser("remove", help="remove a server by name")
    mcp_remove.add_argument("name")
    mcp_remove.add_argument("--scope", choices=MCP_SCOPES, default=MCP_SCOPES[0])
    sub.add_parser("usage", help="token and cost totals per recent session")
    sub.add_parser("skills", help="list discovered skills")
    sub.add_parser("agents", help="list the agent profiles that --agent can select")
    sub.add_parser("commands", help="list the user slash commands from .superclaw/commands and the config dir")
    ctx = sub.add_parser("context", help="show what the first request would cost in context tokens")
    ctx.add_argument("prompt", nargs="?", default="", help="optional prompt, used for memory recall")
    rmap = sub.add_parser("repo-map", help="a deterministic map of the workspace: counts, important files, paths; the same text the model gets")
    rmap.add_argument("--json", action="store_true", help="print the full map as JSON instead of the prompt text")
    rmap.add_argument("--query", default="", metavar="TEXT", help="rank paths against these terms instead of printing the map")
    rmap.add_argument("--max-files", type=int, default=0, help=f"cap the scan (default {LIMITS.repo_map_files})")
    rmap.add_argument("--max-bytes", type=int, default=0, help=f"cap the rendered text (default {LIMITS.repo_map_bytes})")
    plg = sub.add_parser("plugin", help="list, install or remove plugins: directories that bundle skills, agents, commands, hooks and MCP servers")
    plg_sub = plg.add_subparsers(dest="plugin_command")
    plg_sub.add_parser("list", help="plugins found under the workspace and the config dir")
    plg_install = plg_sub.add_parser("install", help="copy a plugin directory, clone a git URL, or fetch <plugin>@<marketplace> into the config dir; superclaw and Claude Code plugin formats")
    plg_install.add_argument("source")
    plg_install.add_argument("--link", action="store_true", help="symlink a local directory instead of copying it, so a checkout stays live")
    plg_sub.add_parser("remove", help="delete an installed plugin by id").add_argument("id")
    market = plg_sub.add_parser("marketplace", help="list, add or remove plugin marketplaces: checkouts with a .claude-plugin/marketplace.json catalogue")
    market_sub = market.add_subparsers(dest="marketplace_command")
    market_sub.add_parser("list", help="known marketplaces and the plugins they offer")
    market_add = market_sub.add_parser("add", help="clone owner/repo or a git URL, or copy a directory, into the config dir")
    market_add.add_argument("source")
    market_add.add_argument("--link", action="store_true", help="symlink a local directory instead of copying it")
    market_sub.add_parser("remove", help="delete a marketplace by name").add_argument("name")
    upd = sub.add_parser("update", help="check for a newer superclaw, and install it with --apply")
    upd.add_argument("--apply", action="store_true", help="run the install command for this install method")
    setup = sub.add_parser("setup", help="store a provider key and default model")
    setup.add_argument("--provider", choices=[p.name for p in PROVIDERS], default=PROVIDERS[0].name)
    setup.add_argument("--key", default="", help="the API key, or the base URL for --provider local; prompted when omitted")
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


def cmd_repo_map(settings: Settings, workspace: Path, args: argparse.Namespace) -> int:
    found = repomap.scan(workspace, max_files=args.max_files or LIMITS.repo_map_files)
    if args.query:
        hits = repomap.search(found, args.query)
        for path, reason in hits:
            print(f"{path:<{LIMITS.model_id_width}} {reason}")
        return 0 if hits else 1
    if args.json:
        print(json.dumps({"root": str(found.root), "fileCount": len(found.files), "directoryCount": found.directories, "truncated": found.truncated,
                          "importantFiles": found.important, "languages": dict(found.languages), "extensions": dict(found.extensions),
                          "files": found.files}, indent=2))
        return 0
    print(repomap.render(found, budget=args.max_bytes or LIMITS.repo_map_bytes))
    return 0


def mcp_file(settings: Settings, workspace: Path, scope: str) -> Path:
    return settings.user_mcp if scope == MCP_SCOPES[0] else workspace / WORKSPACE_DIR / MCP_FILE


def cmd_mcp_edit(settings: Settings, workspace: Path, args: argparse.Namespace) -> int:
    path = mcp_file(settings, workspace, args.scope)
    try:
        if args.mcp_command == "add":
            server = add_server(path, args.name, args.argv, url=args.url, env=args.env, headers=args.header)
            print(f"added {server.name} ({server.transport}) to {path}; `superclaw mcp` connects and lists its tools")
        else:
            remove_server(path, args.name)
            print(f"removed {args.name} from {path}")
    except MCPError as e:
        sys.exit(f"superclaw: {e}")
    return 0


def cmd_marketplace(settings: Settings, args: argparse.Namespace) -> int:
    if args.marketplace_command == "add":
        market = plugins.add_marketplace(args.source, settings.user_marketplaces, link=args.link)
        print(f"added marketplace {market.name} ({len(market.plugins)} plugins) at {market.path}; install one with `superclaw plugin install <plugin>@{market.name}`")
        return 0
    if args.marketplace_command == "remove":
        print(f"removed {plugins.remove_marketplace(args.name, settings.user_marketplaces)}")
        return 0
    found = plugins.load_marketplaces(settings.user_marketplaces)
    for market in found:
        print(f"{market.name:<{_NAME_WIDTH}} {market.description}  ({market.path})")
        print(f"{'':<{_NAME_WIDTH}} {', '.join(sorted(market.plugins)) or 'no plugins'}")
    if not found:
        print(f"no marketplaces; add one with `superclaw plugin marketplace add <owner/repo|git url|dir>` into {settings.user_marketplaces}", file=sys.stderr)
        return 1
    return 0


def cmd_plugin(settings: Settings, workspace: Path, args: argparse.Namespace) -> int:
    try:
        if args.plugin_command == "marketplace":
            return cmd_marketplace(settings, args)
        if args.plugin_command == "install":
            if plugins.is_marketplace_ref(args.source):
                plugin = plugins.install_from_marketplace(args.source, settings.user_marketplaces, settings.user_plugins)
            else:
                plugin = plugins.install(args.source, settings.user_plugins, link=args.link)
            print(f"installed {plugin.id} {plugin.version} ({plugin.format} format{', linked' if args.link else ''}) to {plugin.path}; provides {', '.join(plugin.parts) or 'nothing yet'}")
            return 0
        if args.plugin_command == "remove":
            print(f"removed {plugins.remove(args.id, settings.user_plugins)}")
            return 0
    except plugins.PluginError as e:
        sys.exit(f"superclaw: {e}")
    found = settings.plugins(workspace)
    for plugin in found:
        print(f"{plugin.id:<{_NAME_WIDTH}} {plugin.version:<{_TOKENS_WIDTH}} {plugin.format:<{_TOKENS_WIDTH}} {plugin.description}  [{', '.join(plugin.parts) or 'empty'}]  ({plugin.path})")
    if not found:
        print(f"no plugins; install one with `superclaw plugin install <dir|git url>` into {settings.user_plugins}", file=sys.stderr)
        return 1
    return 0


def cmd_update(settings: Settings, args: argparse.Namespace) -> int:
    found = update.plan(update.detect())
    for line in update.describe(found):
        print(line, file=sys.stderr if args.apply else sys.stdout)
    if not args.apply:
        return 0
    code = update.apply(found)
    if code == 0 and found.available:
        print("superclaw: updated; restart any running session to pick it up", file=sys.stderr)
    return code


def cmd_setup(settings: Settings, args: argparse.Namespace) -> int:
    provider = next(p for p in PROVIDERS if p.name == args.provider)
    if provider.base_env:
        credential = (args.key or input(f"{provider.name} base URL ({provider.console}): ")).strip().rstrip("/")
    else:
        credential = (args.key or getpass.getpass(f"{provider.name} API key ({provider.console}): ")).strip()
    if not credential:
        sys.exit(f"superclaw: no {'URL' if provider.base_env else 'key'} entered")
    os.environ[provider.credential_env] = credential
    model = settings.model if settings.model.startswith(provider.name + "/") or provider.name == PROVIDERS[0].name else provider.default_model
    if not model:
        served = models_for(provider, os.environ.get(provider.env, ""), settings.models_cache, refresh=True)
        model = served[0].id if served else settings.model
    settings.save_credentials(provider, credential, model)
    print(f"saved {provider.credential_env} and SUPERCLAW_MODEL={model} to {settings.credentials}")
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
    agent = None
    if args.agent:
        try:
            agent = resolve_agent(args.agent, defaults.agent_roots(workspace))
        except KeyError as e:
            sys.exit(f"superclaw: {e.args[0]}")
    settings = replace(defaults, model=args.model or (agent.model if agent else "") or defaults.model,
                       mode=mode, context_window=args.context_window,
                       fallback_models=split_models(args.fallback_model) or defaults.fallback_models,
                       budget_tokens=args.budget_tokens, budget_usd=args.budget_usd, db_path=Path(args.db))
    if args.command == "setup":
        return cmd_setup(settings, args)
    if args.command == "models":
        return cmd_models(settings, args)
    if args.command == "update":
        return cmd_update(settings, args)
    if args.command == "plugin":
        return cmd_plugin(settings, workspace, args)
    if args.command == "mcp" and args.mcp_command in ("add", "remove"):
        return cmd_mcp_edit(settings, workspace, args)
    if args.command == "repo-map":
        return cmd_repo_map(settings, workspace, args)
    if args.worktree is not None:
        try:
            tree = prepare_worktree(workspace, Path(args.worktree_dir) if args.worktree_dir else settings.worktrees_dir, args.worktree)
        except WorktreeError as e:
            sys.exit(f"superclaw: {e}")
        print(f"superclaw: {'reusing' if tree.reused else 'created'} worktree {tree.path} on {tree.branch}", file=sys.stderr)
        workspace = tree.path
    try:
        rt = build_runtime(settings, workspace, Mode(mode), max_turns=args.max_turns, intent_gate=args.intent_gate,
                           hooks=build_hooks(settings, workspace, args.trust_workspace),
                           require_provider=args.command not in (None, *STORELESS) and not (args.command == "verify" and args.attempts <= 1)
                           and not (args.command == "spec" and args.spec_command != "approve")
                           and not (args.command == "cron" and args.cron_command not in ("run", "add"))
                           and not (args.command == "mcp" and args.mcp_command == "serve"),
                           open_store=(args.command not in STORELESS or (args.command == "mcp" and args.mcp_command == "serve"))
                           and not (args.command == "verify" and args.attempts <= 1)
                           and not (args.command == "spec" and args.spec_command != "approve"),
                           allow_tools=_tool_set(args.allow_tools), deny_tools=_tool_set(args.deny_tools), extra_dirs=extra_dirs,
                           mcp_config=mcp_paths(settings, workspace, args.trust_workspace), agent=agent)
    except NoProviderKey as e:
        sys.exit(f"superclaw: {e}")
    except StoreInUse as e:
        remedy = "" if isinstance(e, NotServing) else "\n  close the other superclaw, or give this one its own store with --db <path>"
        sys.exit(f"superclaw: {e}{remedy}")
    handler = {"exec": cmd_exec, "acp": cmd_acp, "verify": cmd_verify, "spec": cmd_spec, "cron": cmd_cron, "review": cmd_review,
               "sessions": cmd_sessions, "facts": cmd_facts, "ask": cmd_ask, "maintain": cmd_maintain, "export": cmd_export, "import": cmd_import,
               "usage": cmd_usage, "skills": cmd_skills, "agents": cmd_agents, "commands": cmd_commands,
               "context": cmd_context, "doctor": cmd_doctor, "mcp": cmd_mcp}.get(args.command, cmd_tui)
    try:
        return handler(rt, args)
    except KeyError as e:
        sys.exit(f"superclaw: {e.args[0]}")
    except RuntimeError as e:
        advice = hint(str(e), tui=False)
        sys.exit(f"superclaw: {e}" + (f"\n  {advice}" if advice else ""))
    finally:
        rt.close(SESSION_END_EXIT if args.command is None else SESSION_END_OTHER)


if __name__ == "__main__":
    sys.exit(main())
