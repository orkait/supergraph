from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from superclaw import maintain
from superclaw.catalog import describe, keyed_providers
from superclaw.config import current as current_value
from superclaw.config import find as find_option
from superclaw.config import set_option
from superclaw.facts import as_of_ms
from superclaw.policy import Mode
from superclaw.prompt import _git_branch
from superclaw.settings import SANDBOX_OFF, SANDBOX_ON, SANDBOX_STATES
from superclaw.tui.config import ConfigScreen
from superclaw.usercommands import UserCommand, load_commands

if TYPE_CHECKING:
    from superclaw.tui.app import SuperclawApp


@dataclass(frozen=True)
class Command:
    name: str
    usage: str
    help: str
    run: Callable[[SuperclawApp, str], None]
    aliases: tuple[str, ...] = ()

    def answers_to(self, name: str) -> bool:
        return name == self.name or name in self.aliases

    def offers(self, head: str) -> bool:
        return any(n.startswith(head) for n in (self.name, *self.aliases))


EXIT_WORDS = ("exit", "quit", ":q", ":q!", ":wq", ":wq!")
SKILL_TAG = "skill:"


def _mode(app: SuperclawApp, arg: str) -> None:
    if arg not in Mode._value2member_map_:
        app.note(f"usage: /mode {'|'.join(m.value for m in Mode)}", error=True)
        return
    app.rt.mode = Mode(arg)
    app.refresh_status()
    app.note(f"mode {arg}")


def _config(app: SuperclawApp, arg: str) -> None:
    key, _, value = arg.partition(" ")
    if not key:
        app.push_screen(ConfigScreen(app.rt))
        return
    option = find_option(key)
    if option is None:
        app.note(f"unknown setting {key!r}; /config lists them", error=True)
        return
    if not value.strip():
        app.note(f"{option.key} {current_value(app.rt.settings, option)} {app.glyphs.dot} {option.label}")
        return
    try:
        told = set_option(app.rt, option.key, value)
    except KeyError as e:
        app.note(str(e.args[0]), error=True)
        return
    app.refresh_status()
    app.note(told)


def _sandbox(app: SuperclawApp, arg: str) -> None:
    if not arg:
        app.note(f"sandbox {SANDBOX_ON if app.rt.policy.sandboxed else SANDBOX_OFF}; usage: /sandbox {'|'.join(SANDBOX_STATES)}")
        return
    _config(app, f"sandbox {arg}")


def _new(app: SuperclawApp, arg: str) -> None:
    app.open_session(app.rt.store.create(cwd=str(app.rt.workspace), model=app.rt.model, branch=_git_branch(app.rt.workspace)))


def _resume(app: SuperclawApp, arg: str) -> None:
    if not arg:
        app.open_resume()
        return
    sid = app.rt.store.latest() if arg == "latest" else arg
    if not sid or app.rt.store.get(sid) is None:
        app.note(f"no session {arg!r}", error=True)
        return
    app.open_session(sid)


def _sessions(app: SuperclawApp, arg: str) -> None:
    if arg.startswith("touching "):
        app.show_tool_result("recall", {"path": arg.removeprefix("touching ").strip()})
        return
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
    app.note(f"this session: {u['calls']} calls {dot} {u['tokens']:,} tokens {dot} {u['cached']:,} cached {dot} ${u['cost_usd']:.4f}")


def _context(app: SuperclawApp, arg: str) -> None:
    from superclaw.report import context_report

    def measure() -> list[str]:
        report = context_report(app.rt, arg)
        return [f"{name:<18}{tokens:>9,}  {report.percent(tokens):5.1f}%" for name, tokens in [*report.categories.items(), ("free", report.free)]]

    app.defer("measuring context", measure)


def _recall(app: SuperclawApp, arg: str) -> None:
    if not arg:
        app.note("usage: /recall <§id> | <query>", error=True)
        return
    args = {"ref": arg} if arg.lstrip("§").isalnum() and len(arg.lstrip("§")) >= app.limits.ref_hex_chars else {"query": arg}
    app.show_tool_result("recall", args)


def _facts(app: SuperclawApp, arg: str) -> None:
    facts = app.rt.memory.facts
    words = arg.split()
    if words[:1] == ["retract"] and len(words) >= 2:
        facts.retract(words[1], " ".join(words[2:]) or "retracted by the user")
        app.note(f"retracted {words[1]}")
        return
    as_of = None
    if words[:1] == ["as-of"] and len(words) >= 2:
        as_of = as_of_ms(words[1])
        if as_of is None:
            app.note("usage: /facts as-of YYYY-MM-DD [query]", error=True)
            return
        words = words[2:]
    query = " ".join(words)
    found = facts.search(query, as_of=as_of) if query else facts.recent(as_of=as_of)
    for fact in found:
        app.note(f"{fact.id}  {fact.line()}")
    if not found:
        app.note("no facts yet; web_fetch with a prompt learns them, memory_note with origin web files one")


def _maintain(app: SuperclawApp, arg: str) -> None:
    def sweep() -> list[str]:
        report = maintain.maintain(app.rt.gs)
        return [f"maintained: {report.line(app.glyphs.dot)}", maintain.health_line(app.rt.gs, app.glyphs.dot)]

    app.defer("maintaining the brain", sweep)


def _snapshots(app: SuperclawApp, arg: str) -> None:
    names = maintain.snapshots(app.rt.gs)
    for name in names:
        app.note(name)
    if not names:
        app.note("no snapshots in this process; one is taken before the first unsafe run of a session")


def _rollback(app: SuperclawApp, arg: str) -> None:
    if not arg or arg not in maintain.snapshots(app.rt.gs):
        app.note("usage: /rollback <name>, one of /snapshots", error=True)
        return
    maintain.rollback(app.rt.gs, arg)
    app.open_session(app.rt.store.create(cwd=str(app.rt.workspace), model=app.rt.model, branch=_git_branch(app.rt.workspace)))
    app.note(f"rolled back to {arg}; everything written after it is gone, and this is a fresh session on the restored store")


def _ask(app: SuperclawApp, arg: str) -> None:
    if not arg:
        app.note("usage: /ask <question>", error=True)
        return
    def consult() -> list[str]:
        answer = app.rt.memory.facts.ask(arg)
        lines = [answer.text or "no information available"]
        if answer.cited:
            lines.append("cited: " + ", ".join(answer.cited))
        return lines

    app.defer("asking the substrate", consult)


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

    app.defer("checking health", lambda: list(doctor_lines(app.rt, "/setup")))


def _effort(app: SuperclawApp, arg: str) -> None:
    app.set_effort(arg.strip())


def _tui(app: SuperclawApp, arg: str) -> None:
    app.set_renderer(arg.strip().lower())


def _help(app: SuperclawApp, arg: str) -> None:
    for command in (*COMMANDS, *app.user_commands):
        app.note(f"{command.usage:<24} {command.help}")
    app.note(f" {app.glyphs.dot} ".join(("up/down and tab pick a command", "esc cancels the run", "ctrl+c quits", f"so does a bare {', '.join(EXIT_WORDS)}", "click a card to expand it")))


def _setup(app: SuperclawApp, arg: str) -> None:
    app.open_setup()


def _quit(app: SuperclawApp, arg: str) -> None:
    app.exit()


COMMANDS = (
    Command("/mode", "/mode ask|auto|plan|unsafe", "switch the permission mode", _mode),
    Command("/model", "/model [list|id]", "show or switch the active model", _model),
    Command("/effort", "/effort low|medium|high|off", "set the model's reasoning effort", _effort),
    Command("/tui", "/tui default|fullscreen", "pick the renderer for the next launch and save it", _tui),
    Command("/config", "/config [key [value]]", "every setting in one place: bare opens the editor, or set one by name", _config),
    Command("/sandbox", f"/sandbox [{'|'.join(SANDBOX_STATES)}]", "run bash inside the sandbox or on the host, now and on the next launch", _sandbox),
    Command("/new", "/new, /clear, /reset", "start a fresh session with an empty context; this one stays resumable", _new, aliases=("/clear", "/reset")),
    Command("/resume", "/resume [id|latest]", "pick an earlier session to continue, or name one", _resume),
    Command("/sessions", "/sessions [query|touching <path>]", "list recent sessions, search their events, or see who touched a file", _sessions),
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
    Command("/facts", "/facts [query|as-of DATE [query]|retract ID [reason]]", "facts learned from sources, with age and URL", _facts),
    Command("/ask", "/ask <question>", "answer from stored facts and results with the substrate's reader, no agent loop", _ask),
    Command("/maintain", "/maintain", "expire, decay stale facts, optimize the brain and show its health", _maintain),
    Command("/snapshots", "/snapshots", "snapshots taken in this process", _snapshots),
    Command("/rollback", "/rollback <name>", "restore the store to a snapshot from this process, discarding later writes", _rollback),
    Command("/setup", "/setup", "connect a provider key and model", _setup),
    Command("/help", "/help", "commands and keys", _help),
    Command("/exit", "/exit, /quit", "leave superclaw", _quit, aliases=("/quit",)),
)


def user_entries(roots: list[Path], skill_roots: list[Path | tuple[Path, str]] | None = None) -> list[Command]:
    def runner(command: UserCommand) -> Callable[[SuperclawApp, str], None]:
        return lambda app, arg: app.run_user_command(command, arg)

    return [Command(f"/{c.name}", f"/{c.name} [args]", f"{SKILL_TAG} {c.description}" if c.skill else c.description, runner(c)) for c in load_commands(roots, skill_roots)]


def matching(prefix: str, extra: Sequence[Command] = ()) -> list[Command]:
    head = prefix.split(maxsplit=1)[0] if prefix.strip() else "/"
    return [c for c in (*COMMANDS, *extra) if c.offers(head)]


def dispatch(app: SuperclawApp, text: str, extra: Sequence[Command] = ()) -> None:
    name, _, arg = text.strip().partition(" ")
    for command in (*COMMANDS, *extra):
        if command.answers_to(name):
            command.run(app, arg.strip())
            return
    app.note(f"unknown command {name}; /help lists them", error=True)
