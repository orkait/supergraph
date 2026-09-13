from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Markdown, OptionList, Static

from superclaw import __version__
from superclaw.app import Callbacks, NoProviderKey, Runtime, run_once, switch_model
from superclaw.catalog import Model, keyed_providers, models_for, provider_of, resolve
from superclaw.loop import Result
from superclaw.prompt import _git_branch
from superclaw.provider import hint
from superclaw.runtime import clip, compact
from superclaw.settings import LIMITS, Glyphs, Provider
from superclaw.tools import ToolContext
from superclaw.tui.cards import ToolCard
from superclaw.tui.commands import dispatch, matching
from superclaw.tui.models import ModelScreen
from superclaw.tui.setup import SetupScreen
from superclaw.tui.status import RunStats, StatusBar, TitleBar, WorkingLine, tier
from superclaw.tui.theme import ACCENT, CSS, MUTED

PROMPT_PLACEHOLDER = "describe a task for superclaw"
WORDMARK = "superclaw"
WORDMARK_ART = (
    "███████╗██╗   ██╗██████╗ ███████╗██████╗  ██████╗██╗      █████╗ ██╗    ██╗",
    "██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██╔════╝██║     ██╔══██╗██║    ██║",
    "███████╗██║   ██║██████╔╝█████╗  ██████╔╝██║     ██║     ███████║██║ █╗ ██║",
    "╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██║     ██║     ██╔══██║██║███╗██║",
    "███████║╚██████╔╝██║     ███████╗██║  ██║╚██████╗███████╗██║  ██║╚███╔███╔╝",
    "╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝",
)
TAGLINE = "Any model. Every tool. A graph for memory."
EXAMPLES = ('Try  "explain this codebase"', '"fix the failing test"', '"add a --json flag"')
HINTS = ("/ commands", "tab complete", "esc cancel", "ctrl+c quit", "click a card to expand")
PHASE_THINKING = "thinking"
PHASE_CANCELLING = "cancelling"
BUSY_COMMANDS = ("/new", "/resume", "/clear", "/model")


def describe(event: dict[str, Any], glyphs: Glyphs) -> str:
    kind = event["type"]
    if kind == "delegate":
        return f"{glyphs.child} delegate {event['child']}: {clip(event['task'], LIMITS.preview_args_chars)}"
    if kind == "prune":
        return f"pruned {event['results']} older results; recall §id brings any back"
    if kind == "compaction":
        return f"compacted {event['removed']} messages into a summary"
    if kind == "permission_decision":
        return f"permission {event['tool']}: {event['decision']}"
    if kind in ("budget", "cancelled", "prompt_drift", "intent", "verdict"):
        return f"{kind}: " + ", ".join(f"{k}={v}" for k, v in event.items() if k not in ("type", "child"))
    return ""


class PermissionScreen(ModalScreen[str]):
    BINDINGS = [
        ("a", "choose('allow')", "Allow once"),
        ("s", "choose('allow_session')", "Allow for session"),
        ("p", "choose('allow_prefix')", "Remember prefix"),
        ("d", "choose('deny')", "Deny"),
        ("escape", "choose('deny')", "Deny"),
        ("ctrl+c", "app.interrupt", "Cancel / quit"),
    ]

    def __init__(self, request: dict[str, Any]) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        args = json.dumps(self.request["args"], indent=1)
        prefix = self.request.get("prefix") or []
        with Vertical(id="dialog"):
            yield Label(f"Permission: {self.request['tool']}", classes="title")
            yield Static(clip(args, LIMITS.dialog_args_chars), classes="args")
            yield Static(f"{self.request['reason']}  {self.app.glyphs.dot}  risk {self.request['risk']} ({', '.join(self.request['categories'])})", classes="reason")
            with Horizontal(classes="buttons"):
                yield Button("Allow once (a)", id="allow", variant="primary")
                yield Button("Allow for session (s)", id="allow_session")
                if prefix:
                    yield Button(f"Remember `{' '.join(prefix)}` (p)", id="allow_prefix")
                yield Button("Deny (d)", id="deny", variant="error")

    def action_choose(self, choice: str) -> None:
        if choice == "allow_prefix" and not self.request.get("prefix"):
            return
        self.dismiss(choice)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id)


class QuestionScreen(ModalScreen[str]):
    BINDINGS = [("escape", "skip", "Skip"), ("ctrl+c", "app.interrupt", "Cancel / quit")]

    def __init__(self, question: dict[str, Any]) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        options = self.question.get("options") or []
        with Vertical(id="dialog"):
            yield Label(self.question["question"], classes="title")
            if options:
                yield OptionList(*options, id="options")
            yield Input(placeholder="type an answer and press enter", id="answer")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(str(event.option.prompt))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.dismiss(event.value.strip())

    def action_skip(self) -> None:
        self.dismiss("")


class SuperclawApp(App[None]):
    CSS = CSS
    BINDINGS = [
        ("ctrl+c", "interrupt", "Cancel / quit"),
        ("escape", "cancel", "Cancel"),
        Binding("down", "palette_move(1)", "Next command", show=False, priority=True),
        Binding("up", "palette_move(-1)", "Previous command", show=False, priority=True),
        Binding("tab", "palette_complete", "Complete command", show=False, priority=True),
    ]
    limits = LIMITS

    def __init__(self, rt: Runtime, session_id: str) -> None:
        self.glyphs = rt.settings.glyphs
        super().__init__()
        self.rt = rt
        self.session_id = session_id
        self.stats = RunStats(window=rt.context_window)
        self.cards: dict[str, ToolCard] = {}
        self.cancel_flag = threading.Event()
        self.running = False
        self.closing = False

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return {"border-kind": self.glyphs.border}

    def compose(self) -> ComposeResult:
        yield TitleBar(id="title")
        yield Static(self.welcome(), id="welcome")
        yield VerticalScroll(id="transcript", can_focus=False)
        yield Static(f" {self.glyphs.dot} ".join(HINTS), id="hints")
        yield WorkingLine()
        with Horizontal(id="composer"):
            yield Static(self.glyphs.prompt, classes="gutter")
            yield Input(placeholder=PROMPT_PLACEHOLDER, id="prompt", select_on_focus=False)
        yield StatusBar(id="status")
        yield OptionList(id="palette", classes="hidden")

    def on_mount(self) -> None:
        self.query_one("#transcript").display = False
        self.refresh_status()
        self.set_interval(LIMITS.spinner_interval_s, self.tick)
        self.query_one("#prompt", Input).focus()
        if self.rt.provider is None:
            self.open_setup()

    def open_setup(self, provider: Provider | None = None) -> None:
        self.push_screen(SetupScreen(self.rt, provider), self.after_setup)

    def after_setup(self, connected: bool | None) -> None:
        self.stats = RunStats(window=self.rt.context_window)
        self.refresh_status()
        self.query_one("#welcome", Static).update(self.welcome())
        if connected:
            self.note(f"model {self.rt.model} {self.glyphs.dot} {compact(self.rt.context_window)} window {self.glyphs.dot} /model switches")
        else:
            self.note("no provider connected; /setup when you have a key", error=True)

    def known_models(self) -> list[Model]:
        return [m for p in keyed_providers() for m in models_for(p, os.environ.get(p.env, ""), self.rt.settings.models_cache, online=False)]

    def recent_models(self) -> list[str]:
        seen = dict.fromkeys([self.rt.model, *(s["model"] for s in self.rt.store.recent())])
        return list(seen)[: LIMITS.recent_models_shown]

    def open_models(self) -> None:
        providers = keyed_providers()
        if not providers:
            self.open_setup()
            return
        self.push_screen(ModelScreen(self.rt, providers, self.rt.model, self.recent_models()), self.after_model)

    def after_model(self, model: str | None) -> None:
        if model:
            self.switch_model(model)

    def switch_model(self, text: str) -> None:
        match = resolve(text, self.known_models(), self.rt.model.split("/")[0])
        target = match.id if match else text.strip()
        if provider_of(target) is None:
            self.note(f"no model matches {text!r}; /model lists them, or name one as provider/id", error=True)
            return
        try:
            switch_model(self.rt, target)
        except NoProviderKey as e:
            self.note(f"{e}; connect it first", error=True)
            self.open_setup(provider_of(target))
            return
        self.stats.window = self.rt.context_window
        self.refresh_status()
        self.note(f"model {target} {self.glyphs.dot} {compact(self.rt.context_window)} window")

    def relayout(self) -> None:
        self.refresh_status()
        self.query_one("#welcome", Static).update(self.welcome())

    def welcome(self) -> Text:
        width = self.size.width or LIMITS.tui_tier_full
        parts = (f"v{__version__}", self.short_cwd(width), _git_branch(self.rt.workspace), self.rt.model)
        sep = f"  {self.glyphs.dot}  "
        mark = [Text(row, style=ACCENT) for row in WORDMARK_ART] if tier(width) >= 2 and self.glyphs.block_art else [Text(WORDMARK, style=ACCENT)]
        lines = [*mark, Text(""), Text(TAGLINE, style=MUTED), Text(""), Text(sep.join(p for p in parts if p), style=MUTED), Text("")]
        if tier(width) >= 2:
            lines += [Text(sep.join(EXAMPLES), style=MUTED), Text("")]
        return Text("\n").join(lines)

    def refresh_status(self) -> None:
        width = self.size.width
        self.query_one("#title", TitleBar).show(self.short_cwd(width), _git_branch(self.rt.workspace), self.session_id, width)
        self.query_one("#composer", Horizontal).border_subtitle = self.rt.model if tier(width) >= 1 else ""
        self.query_one("#status", StatusBar).show(self.rt.mode.value, self.stats, width)

    def short_cwd(self, width: int) -> str:
        return clip(str(self.rt.workspace).replace(str(Path.home()), "~", 1), max(LIMITS.card_arg_chars // 2, width // 3))

    def tick(self) -> None:
        if self.running:
            self.query_one(WorkingLine).tick(self.stats.timer.elapsed(), self.stats.timer.calls)

    def transcript(self) -> VerticalScroll:
        view = self.query_one("#transcript", VerticalScroll)
        if not view.display:
            self.query_one("#welcome").display = False
            view.display = True
        return view

    def add(self, widget: Static | Markdown | ToolCard) -> None:
        view = self.transcript()
        view.mount(widget)
        view.scroll_end(animate=False)

    def note(self, text: str, error: bool = False) -> None:
        self.add(Static(text, classes="error" if error else "note"))

    def clear_transcript(self) -> None:
        self.query_one("#transcript", VerticalScroll).remove_children()
        self.cards.clear()
        self.query_one("#transcript").display = False
        self.query_one("#welcome").display = True

    def open_session(self, sid: str) -> None:
        self.session_id = sid
        self.clear_transcript()
        self.stats = RunStats(window=self.rt.context_window)
        self.refresh_status()

    def show_tool_result(self, name: str, args: dict[str, Any]) -> None:
        card = ToolCard("local", name, args)
        self.add(card)
        res = self.rt.registry.run(name, args, ToolContext(workspace=self.rt.workspace, session_id=self.session_id))
        card.finish(res.ok, res.output, {}, res.artifact.ref if res.artifact else "")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt":
            return
        palette = self.query_one("#palette", OptionList)
        text = event.value
        if text.startswith("/") and " " not in text:
            palette.clear_options()
            for command in matching(text)[: LIMITS.command_matches_shown]:
                palette.add_option(f"{command.usage:<26} {command.help}")
            palette.highlighted = None
            palette.remove_class("hidden") if palette.option_count else palette.add_class("hidden")
        else:
            palette.add_class("hidden")

    def palette_open(self) -> bool:
        return not self.query_one("#palette", OptionList).has_class("hidden")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        return self.palette_open() if action.startswith("palette_") else True

    def action_palette_move(self, step: int) -> None:
        palette = self.query_one("#palette", OptionList)
        current = -1 if palette.highlighted is None and step > 0 else (palette.highlighted or 0)
        palette.highlighted = (current + step) % palette.option_count

    def action_palette_complete(self) -> None:
        palette = self.query_one("#palette", OptionList)
        self.pick(palette.highlighted or 0)

    def pick(self, index: int) -> None:
        palette = self.query_one("#palette", OptionList)
        prompt = self.query_one("#prompt", Input)
        usage = str(palette.get_option_at_index(index).prompt).split("  ")[0].strip()
        name = usage.split()[0]
        palette.add_class("hidden")
        prompt.focus()
        if usage == name:
            prompt.value = ""
            self.command(name)
        else:
            prompt.value = name + " "
            prompt.cursor_position = len(prompt.value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "palette":
            self.pick(event.option_index)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt":
            return
        palette = self.query_one("#palette", OptionList)
        if self.palette_open() and palette.highlighted is not None:
            self.pick(palette.highlighted)
            return
        text = event.value.strip()
        event.input.value = ""
        palette.add_class("hidden")
        if not text:
            return
        if text.startswith("/"):
            self.command(text)
            return
        if self.running:
            self.note("a run is in progress; esc cancels it", error=True)
            return
        if self.rt.provider is None:
            self.open_setup()
            return
        self.add(Static(f"{self.glyphs.prompt} {text}", classes="user"))
        self.begin_run(text)

    def command(self, text: str) -> None:
        name = text.split()[0]
        if self.running and name in BUSY_COMMANDS:
            self.note(f"{name} waits for the run to finish; esc cancels it", error=True)
        else:
            dispatch(self, text)

    def begin_run(self, text: str) -> None:
        self.running = True
        self.cancel_flag.clear()
        self.stats.timer.start()
        self.query_one("#hints").add_class("hidden")
        self.query_one(WorkingLine).start(PHASE_THINKING)
        self.query_one("#prompt", Input).placeholder = "esc to cancel"
        self.run_prompt(text)

    def action_cancel(self) -> None:
        palette = self.query_one("#palette", OptionList)
        if not palette.has_class("hidden"):
            palette.add_class("hidden")
            self.query_one("#prompt", Input).value = ""
        elif self.running:
            self.cancel_flag.set()
            self.query_one(WorkingLine).start(PHASE_CANCELLING)

    def action_interrupt(self) -> None:
        if self.running and not self.cancel_flag.is_set():
            self.action_cancel()
            self.note("cancelling; ctrl+c again to quit", error=True)
        else:
            self.exit()

    @work(thread=True, exclusive=True)
    def run_prompt(self, text: str) -> None:
        callbacks = Callbacks(on_event=lambda event: self.call_from_thread(self.render_event, event),
                              on_permission=self.ask_permission, on_ask_user=self.ask_questions)
        try:
            result = run_once(self.rt, text, self.session_id, callbacks, cancelled=self.cancel_flag.is_set)
        except Exception as e:
            self.call_from_thread(self.finish, None, clip(str(e), LIMITS.preview_error_chars))
            return
        self.call_from_thread(self.finish, result)

    def render_event(self, event: dict[str, Any]) -> None:
        kind, child = event["type"], bool(event.get("child"))
        if kind == "text" and not child:
            self.add(Markdown(event["text"]))
        elif kind in ("tool_call", "tool_result"):
            self.render_tool(event, child)
        elif kind == "usage" and not child:
            self.stats.used, self.stats.window = event["context_used"], event["context_window"]
            self.stats.tokens, self.stats.cost = event["run_total"], event["run_cost_usd"]
            self.stats.saved, self.stats.kept_out = event["saved_tokens"], event["kept_out_tokens"]
            self.refresh_status()
        elif line := describe(event, self.glyphs):
            self.note(line)

    def render_tool(self, event: dict[str, Any], child: bool) -> None:
        key = f"{event.get('child', '')}:{event['id']}"
        working = self.query_one(WorkingLine)
        if event["type"] == "tool_call":
            card = ToolCard(event["id"], event["name"], event["args"], child=child)
            self.cards[key] = card
            self.add(card)
            self.stats.timer.calls += 1
            working.start(event["name"])
            return
        card = self.cards.pop(key, None)
        if card:
            card.finish(event["ok"], event["output"], event.get("display") or {}, event.get("ref", ""))
        working.start(PHASE_THINKING)

    def finish(self, result: Result | None, error: str = "") -> None:
        self.running = False
        self.query_one(WorkingLine).stop()
        self.query_one("#hints").remove_class("hidden")
        elapsed = self.stats.timer.elapsed()
        sep = f" {self.glyphs.dot} "
        if result is None:
            advice = hint(error, tui=True)
            self.note(f"run failed after {elapsed:.0f}s: {error}" + (f"{sep}{advice}" if advice else ""), error=True)
        else:
            summary = sep.join((f"done in {elapsed:.0f}s", f"{result.turns} turns", f"{self.stats.timer.calls} tools",
                                f"{result.saved_tokens + result.kept_out_tokens:,} tokens kept out of the window"))
            if result.stop_reason or result.incomplete:
                self.note(f"stopped: {result.stop_reason or result.incomplete_reason}{sep}{summary}", error=True)
            else:
                self.note(summary)
        prompt = self.query_one("#prompt", Input)
        prompt.placeholder = PROMPT_PLACEHOLDER
        prompt.focus()
        self.refresh_status()

    def modal(self, screen: ModalScreen[str]) -> str:
        done = threading.Event()
        answer: dict[str, str] = {}

        def settle(value: str | None) -> None:
            answer["value"] = value or ""
            done.set()

        self.stats.timer.pause()
        self.call_from_thread(self.push_screen, screen, settle)
        while not done.wait(LIMITS.timer_interval_s):
            if self.closing:
                return ""
            if self.cancel_flag.is_set():
                self.call_from_thread(screen.dismiss, None)
        self.stats.timer.resume()
        return answer["value"]

    def on_unmount(self) -> None:
        self.closing = True

    def ask_permission(self, request: dict[str, Any]) -> str:
        return self.modal(PermissionScreen(request)) or "deny"

    def ask_questions(self, questions: list[dict[str, Any]]) -> list[str]:
        return [self.modal(QuestionScreen(q)) for q in questions]
