from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from superclaw.catalog import describe, keyed_providers
from superclaw.policy import Mode

if TYPE_CHECKING:
    from superclaw.tui.app import SuperclawApp


@dataclass(frozen=True)
class Command:
    name: str
    usage: str
    help: str
    run: Callable[[SuperclawApp, str], None]


def _mode(app: SuperclawApp, arg: str) -> None:
    if arg not in Mode._value2member_map_:
        app.note(f"usage: /mode {'|'.join(m.value for m in Mode)}", error=True)
        return
    app.rt.mode = Mode(arg)
    app.refresh_status()
    app.note(f"mode {arg}")


def _new(app: SuperclawApp, arg: str) -> None:
    app.open_session(app.rt.store.create(cwd=str(app.rt.workspace), model=app.rt.model))


def _resume(app: SuperclawApp, arg: str) -> None:
    sid = app.rt.store.latest() if arg in ("", "latest") else arg
    if not sid or app.rt.store.get(sid) is None:
        app.note(f"no session {arg or 'latest'!r}", error=True)
        return
    app.open_session(sid)


def _sessions(app: SuperclawApp, arg: str) -> None:
    if arg:
        hits = app.rt.store.search(arg)
        if not hits:
            app.note(f"no session event matches {arg!r}", error=True)
        for hit in hits:
            app.note(f"{hit['id']}  #{hit['seq']} {hit['type']}  {hit['text']}")
        return
    for s in app.rt.store.recent()[: app.limits.recent_sessions_shown]:
        app.note(f"{s['id']}  {s['event_count']} events  {s['cwd']}")


def _fork(app: SuperclawApp, arg: str) -> None:
    source = app.session_id if arg == "" else (app.rt.store.latest() if arg == "latest" else arg)
    if not source or app.rt.store.get(source) is None:
        app.note(f"no session {arg or 'latest'!r}", error=True)
        return
    app.open_session(app.rt.store.fork(source))
    app.note(f"forked {source} into {app.session_id}")


def _usage(app: SuperclawApp, arg: str) -> None:
    u = app.rt.store.usage(app.session_id)
    dot = app.glyphs.dot
    app.note(f"this session: {u['calls']} calls {dot} {u['tokens']:,} tokens {dot} ${u['cost_usd']:.4f}")


def _context(app: SuperclawApp, arg: str) -> None:
    from superclaw.report import context_report

    report = context_report(app.rt, arg)
    for name, tokens in [*report.categories.items(), ("free", report.free)]:
        app.note(f"{name:<18}{tokens:>9,}  {report.percent(tokens):5.1f}%")


def _recall(app: SuperclawApp, arg: str) -> None:
    if not arg:
        app.note("usage: /recall <§id> | <query>", error=True)
        return
    args = {"ref": arg} if arg.lstrip("§").isalnum() and len(arg.lstrip("§")) >= app.limits.ref_hex_chars else {"query": arg}
    app.show_tool_result("recall", args)


def _model(app: SuperclawApp, arg: str) -> None:
    if not arg:
        app.open_models()
    elif arg == "list":
        for provider in keyed_providers():
            models = [m for m in app.known_models() if m.provider == provider.name]
            for model in models[: app.limits.model_list_shown]:
                mark = app.glyphs.prompt if model.id == app.rt.model else " "
                app.note(f"{mark} {model.id:<{app.limits.model_id_width}} {describe(model, app.glyphs.dot)}")
            if len(models) > app.limits.model_list_shown:
                app.note(f"  {app.glyphs.ellipsis} {len(models) - app.limits.model_list_shown} more on {provider.name}; /model searches them")
    else:
        app.switch_model(arg)


def _compact(app: SuperclawApp, arg: str) -> None:
    app.do_compact()


def _retry(app: SuperclawApp, arg: str) -> None:
    app.retry()


def _rename(app: SuperclawApp, arg: str) -> None:
    app.rename(arg.strip())


def _export(app: SuperclawApp, arg: str) -> None:
    app.export()


def _tools(app: SuperclawApp, arg: str) -> None:
    visible = app.rt.policy.visible
    for tool in app.rt.registry.tools():
        mark = " " if visible(tool) else app.glyphs.failed
        kind = "lazy" if tool.deferred else "eager"
        app.note(f"{mark} {tool.name:<{app.limits.tool_name_width}} {tool.safety.side_effect.value} {app.glyphs.dot} {kind}")


def _agent(app: SuperclawApp, arg: str) -> None:
    app.use_agent(arg.strip())


def _attach(app: SuperclawApp, arg: str) -> None:
    app.attach(arg.strip())


def _mcp(app: SuperclawApp, arg: str) -> None:
    bridge = app.rt.mcp
    for tool in bridge.tools if bridge else []:
        app.note(f"{tool.server} {app.glyphs.dot} {tool.name}: {tool.summary()}")
    for skipped in bridge.skipped if bridge else []:
        app.note(f"{skipped.name} skipped: {skipped.error}", error=True)
    for problem in bridge.problems if bridge else []:
        app.note(f"config: {problem}", error=True)
    if bridge is None or not bridge.tools:
        app.note(f"no MCP tools; add servers to {app.rt.settings.user_mcp}", error=True)


def _permissions(app: SuperclawApp, arg: str) -> None:
    dot = app.glyphs.dot
    app.note(f"mode {app.rt.mode.value}")
    app.note("session grants: " + (f" {dot} ".join(app.rt.policy.session_grants) or "none"))
    app.note("remembered prefixes: " + (f" {dot} ".join(" ".join(p) for p in app.rt.policy.prefix_grants) or "none"))


def _doctor(app: SuperclawApp, arg: str) -> None:
    from superclaw.report import doctor_lines

    for line in doctor_lines(app.rt, "/setup"):
        app.note(line)


def _effort(app: SuperclawApp, arg: str) -> None:
    app.set_effort(arg.strip())


def _clear(app: SuperclawApp, arg: str) -> None:
    app.clear_transcript()


def _help(app: SuperclawApp, arg: str) -> None:
    for command in COMMANDS:
        app.note(f"{command.usage:<24} {command.help}")
    app.note(f" {app.glyphs.dot} ".join(("up/down and tab pick a command", "esc cancels the run", "ctrl+c quits", "click a card to expand it")))


def _setup(app: SuperclawApp, arg: str) -> None:
    app.open_setup()


def _quit(app: SuperclawApp, arg: str) -> None:
    app.exit()


COMMANDS = (
    Command("/mode", "/mode ask|auto|plan|unsafe", "switch the permission mode", _mode),
    Command("/model", "/model [list|id]", "show or switch the active model", _model),
    Command("/effort", "/effort low|medium|high|off", "set the model's reasoning effort", _effort),
    Command("/new", "/new", "start a fresh session", _new),
    Command("/resume", "/resume [id|latest]", "continue an earlier session", _resume),
    Command("/sessions", "/sessions [query]", "list recent sessions, or search their events", _sessions),
    Command("/fork", "/fork [id|latest]", "copy a session into a new one and continue it", _fork),
    Command("/usage", "/usage", "tokens and cost spent in this session", _usage),
    Command("/context", "/context [prompt]", "what the next request costs", _context),
    Command("/compact", "/compact", "summarize older turns to free the window now", _compact),
    Command("/retry", "/retry", "run the last prompt again", _retry),
    Command("/rename", "/rename <title>", "name this session", _rename),
    Command("/export", "/export", "write the transcript to a markdown file", _export),
    Command("/tools", "/tools", "list the tools and their side effects", _tools),
    Command("/agent", "/agent [name|none]", "show or switch the agent profile", _agent),
    Command("/attach", "/attach <path>", "attach a file or image to your next message", _attach),
    Command("/mcp", "/mcp", "list the MCP servers and the tools they expose", _mcp),
    Command("/permissions", "/permissions", "show the mode and remembered grants", _permissions),
    Command("/doctor", "/doctor", "terminal, sandbox, model and provider health", _doctor),
    Command("/recall", "/recall <§id|query>", "bring back or search stored tool results", _recall),
    Command("/clear", "/clear", "clear the transcript view", _clear),
    Command("/setup", "/setup", "connect a provider key and model", _setup),
    Command("/help", "/help", "commands and keys", _help),
    Command("/quit", "/quit", "exit", _quit),
)


def matching(prefix: str) -> list[Command]:
    head = prefix.split()[0] if prefix.strip() else "/"
    return [c for c in COMMANDS if c.name.startswith(head)]


def dispatch(app: SuperclawApp, text: str) -> None:
    name, _, arg = text.strip().partition(" ")
    for command in COMMANDS:
        if command.name == name:
            command.run(app, arg.strip())
            return
    app.note(f"unknown command {name}; /help lists them", error=True)
