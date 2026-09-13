from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Markdown, OptionList, Static

from superclaw import __version__
from superclaw.app import Callbacks, Runtime, run_once
from superclaw.loop import Result
from superclaw.prompt import _git_branch
from superclaw.runtime import clip
from superclaw.settings import LIMITS
from superclaw.tools import ToolContext
from superclaw.tui.cards import ToolCard
from superclaw.tui.commands import dispatch, matching
from superclaw.tui.setup import SetupScreen
from superclaw.tui.status import RunStats, StatusBar, TitleBar, WorkingLine, tier
from superclaw.tui.theme import ACCENT, CSS, MUTED

PROMPT_PLACEHOLDER = "describe a task for superclaw…"
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
EXAMPLES = 'Try  "explain this codebase"  ·  "fix the failing test"  ·  "add a --json flag"'
HINT = "/ commands · esc cancel · ctrl+c quit · click a card to expand"
PHASE_THINKING = "thinking"
PHASE_CANCELLING = "cancelling"
BUSY_COMMANDS = ("/new", "/resume", "/clear")


def describe(event: dict[str, Any]) -> str:
    kind = event["type"]
    if kind == "delegate":
        return f"↳ delegate {event['child']}: {clip(event['task'], LIMITS.preview_args_chars)}"
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
            yield Static(f"{self.request['reason']}  ·  risk {self.request['risk']} ({', '.join(self.request['categories'])})", classes="reason")
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
    BINDINGS = [("escape", "skip", "Skip")]

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
    BINDINGS = [("ctrl+c", "interrupt", "Cancel / quit"), ("escape", "cancel", "Cancel")]
    limits = LIMITS

    def __init__(self, rt: Runtime, session_id: str) -> None:
        super().__init__()
        self.rt = rt
        self.session_id = session_id
        self.stats = RunStats(window=rt.context_window)
        self.cards: dict[str, ToolCard] = {}
        self.cancel_flag = threading.Event()
        self.running = False

    def compose(self) -> ComposeResult:
        yield TitleBar(id="title")
        yield Static(self.welcome(), id="welcome")
        yield VerticalScroll(id="transcript")
        yield Static(HINT, id="hints")
        yield WorkingLine()
        with Horizontal(id="composer"):
            yield Static("❯", classes="gutter")
            yield Input(placeholder=PROMPT_PLACEHOLDER, id="prompt")
        yield StatusBar(id="status")
        yield OptionList(id="palette", classes="hidden")

    def on_mount(self) -> None:
        self.query_one("#transcript").display = False
        self.refresh_status()
        self.set_interval(LIMITS.spinner_interval_s, self.tick)
        self.query_one("#prompt", Input).focus()
        if self.rt.provider is None:
            self.open_setup()

    def open_setup(self) -> None:
        self.push_screen(SetupScreen(self.rt), self.after_setup)

    def after_setup(self, connected: bool | None) -> None:
        self.stats = RunStats(window=self.rt.context_window)
        self.refresh_status()
        self.query_one("#welcome", Static).update(self.welcome())
        if not connected:
            self.note("no provider connected; /setup when you have a key", error=True)

    def on_resize(self) -> None:
        self.refresh_status()
        self.query_one("#welcome", Static).update(self.welcome())

    def welcome(self) -> Text:
        width = self.size.width or LIMITS.tui_tier_full
        parts = (f"v{__version__}", self.short_cwd(width), _git_branch(self.rt.workspace), self.rt.model)
        mark = [Text(row, style=ACCENT) for row in WORDMARK_ART] if tier(width) >= 2 else [Text(WORDMARK, style=ACCENT)]
        lines = [*mark, Text(""), Text(TAGLINE, style=MUTED), Text(""), Text("  ·  ".join(p for p in parts if p), style=MUTED), Text("")]
        if tier(width) >= 2:
            lines += [Text(EXAMPLES, style=MUTED), Text("")]
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
            palette.remove_class("hidden") if palette.option_count else palette.add_class("hidden")
        else:
            palette.add_class("hidden")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "palette":
            return
        prompt = self.query_one("#prompt", Input)
        usage = str(event.option.prompt).split("  ")[0].strip()
        name = usage.split()[0]
        self.query_one("#palette", OptionList).add_class("hidden")
        prompt.focus()
        if usage == name:
            prompt.value = ""
            dispatch(self, name)
        else:
            prompt.value = name + " "

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt":
            return
        text = event.value.strip()
        event.input.value = ""
        self.query_one("#palette", OptionList).add_class("hidden")
        if not text:
            return
        if text.startswith("/"):
            if self.running and text.split()[0] in BUSY_COMMANDS:
                self.note(f"{text.split()[0]} waits for the run to finish; esc cancels it", error=True)
            else:
                dispatch(self, text)
            return
        if self.running:
            self.note("a run is in progress; esc cancels it", error=True)
            return
        if self.rt.provider is None:
            self.open_setup()
            return
        self.add(Static(f"❯ {text}", classes="user"))
        self.begin_run(text)

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
        result = run_once(self.rt, text, self.session_id, Callbacks(
            on_event=lambda event: self.call_from_thread(self.render_event, event),
            on_permission=self.ask_permission,
            on_ask_user=self.ask_questions,
        ), cancelled=self.cancel_flag.is_set)
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
        elif line := describe(event):
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

    def finish(self, result: Result) -> None:
        self.running = False
        self.query_one(WorkingLine).stop()
        self.query_one("#hints").remove_class("hidden")
        elapsed = self.stats.timer.elapsed()
        summary = f"done in {elapsed:.0f}s · {result.turns} turns · {self.stats.timer.calls} tools · {result.saved_tokens + result.kept_out_tokens:,} tokens kept out of the window"
        if result.stop_reason or result.incomplete:
            self.note(f"stopped: {result.stop_reason or result.incomplete_reason} · {summary}", error=True)
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
        done.wait()
        self.stats.timer.resume()
        return answer["value"]

    def ask_permission(self, request: dict[str, Any]) -> str:
        return self.modal(PermissionScreen(request)) or "deny"

    def ask_questions(self, questions: list[dict[str, Any]]) -> list[str]:
        return [self.modal(QuestionScreen(q)) for q in questions]
