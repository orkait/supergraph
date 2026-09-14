from __future__ import annotations

import json
import os
import threading
import time
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
from superclaw.agents import load_agents
from superclaw.app import Callbacks, NoProviderKey, Runtime, apply_effort, run_once, switch_model
from superclaw.attach import read as read_attachments
from superclaw.catalog import Model, keyed_providers, models_for, provider_of, resolve
from superclaw.loop import Result
from superclaw.policy import next_mode
from superclaw.prompt import _git_branch
from superclaw.provider import hint
from superclaw.runtime import Message, clip, compact, count
from superclaw.compaction import TRANSCRIPT_NOTE, summary_instructions
from superclaw.compaction import compact as compact_context
from superclaw.settings import EFFORT_OFF, EFFORTS, LIMITS, SESSION_END_CLEAR, TRANSCRIPT_TEMPLATE, Glyphs, Provider
from superclaw.tools import ToolContext
from superclaw import clipboard
from superclaw.clips import Clip, Clips
from superclaw.tui.cards import ToolCard, target_of
from superclaw.tui.commands import EXIT_WORDS, dispatch, matching, user_entries
from superclaw.tui.composer import Composer
from superclaw.usercommands import UserCommand, expand
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
HINTS = ("/ commands", "up down history", "shift+tab mode", "ctrl+v paste image", "drop a file to attach", "ctrl+o unfold output", "esc cancel", "ctrl+c quit")
PHASE_RECALLING = "recalling"
PHASE_THINKING = "thinking"
PHASE_WRITING = "writing"
PHASE_WAITING = "waiting for you"
PHASE_CANCELLING = "cancelling"
PHASE_COMPACTING = "compacting"
FRESH_CONTEXT = "fresh context"
BUSY_COMMANDS = ("/new", "/clear", "/reset", "/resume", "/fork", "/model", "/compact", "/retry", "/agent")


def context_overview(event: dict[str, Any], glyphs: Glyphs) -> str:
    parts = [count(event["memories"], "memory", "memories") if event["memories"] else "", count(event["skills"], "skill") if event["skills"] else "",
             f"repo map {count(event['repo_files'], 'file')}" if event["repo_files"] else "", count(event["history"], "earlier message") if event["history"] else ""]
    return f" {glyphs.dot} ".join(p for p in parts if p) or FRESH_CONTEXT


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
    if kind == "prompt_drift":
        return "system prompt changed since this session last ran; the prompt event holds the new one"
    if kind == "snapshot":
        return f"snapshot {event['name']} taken before the first unsafe run of this session; /rollback {event['name']} restores it while this process runs"
    if kind in ("budget", "cancelled", "intent", "verdict"):
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
        Binding("down", "history(1)", "Newer / next command", show=False, priority=True),
        Binding("up", "history(-1)", "Older / previous command", show=False, priority=True),
        Binding("tab", "palette_complete", "Complete command", show=False, priority=True),
        Binding("shift+tab", "cycle_mode", "Cycle mode", show=False, priority=True),
        Binding("ctrl+o", "toggle_verbose", "Unfold tool output", show=False, priority=True),
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
        self.history: list[str] = []
        self.hist_index = 0
        self.hist_draft = ""
        self._title = ""
        self.clips = Clips()
        self.user_commands = user_entries(rt.settings.command_roots(rt.workspace), rt.settings.skill_roots(rt.workspace))
        self.streaming: Static | None = None
        self.stream_text = ""
        self.verbose = False
        self.current_tool = ("", "")

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
            yield Composer(placeholder=PROMPT_PLACEHOLDER, id="prompt", select_on_focus=False)
        yield StatusBar(id="status")
        yield OptionList(id="palette", classes="hidden")

    def on_mount(self) -> None:
        self.query_one("#transcript").display = False
        self.refresh_status()
        self.load_history()
        self.set_interval(LIMITS.spinner_interval_s, self.tick)
        self.query_one("#prompt", Input).focus()
        if self.rt.provider is None:
            self.open_setup()

    def load_meta(self) -> None:
        self._title = (self.rt.store.get(self.session_id) or {}).get("title", "")

    def session_label(self) -> str:
        return self._title or self.session_id

    def load_history(self) -> None:
        self.load_meta()
        prompts, expect = [], False
        for event in self.rt.store.events(self.session_id):
            if event["type"] == "prompt":
                expect = True
            elif expect and event["type"] == "message" and event["payload"].get("role") == "user":
                prompts.append(event["payload"].get("content", ""))
                expect = False
        self.history = [p for p in prompts if p.strip()]
        self.reset_history()

    def reset_history(self) -> None:
        self.hist_index = len(self.history)
        self.hist_draft = ""

    def remember(self, text: str) -> None:
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self.reset_history()

    def action_history(self, step: int) -> None:
        if self.palette_open():
            self.action_palette_move(step)
            return
        prompt = self.query_one("#prompt", Input)
        if not self.history or (self.hist_index == len(self.history) and step > 0):
            return
        if self.hist_index == len(self.history):
            self.hist_draft = prompt.value
        self.hist_index = max(0, min(len(self.history), self.hist_index + step))
        prompt.value = self.hist_draft if self.hist_index == len(self.history) else self.history[self.hist_index]
        prompt.cursor_position = len(prompt.value)

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
        self.query_one("#title", TitleBar).show(self.short_cwd(width), _git_branch(self.rt.workspace), self.session_label(), width)
        self.query_one("#composer", Horizontal).border_subtitle = self.rt.model if tier(width) >= 1 else ""
        self.query_one("#status", StatusBar).show(self.rt.mode.value, self.stats, width)

    def short_cwd(self, width: int) -> str:
        return clip(str(self.rt.workspace).replace(str(Path.home()), "~", 1), max(LIMITS.card_arg_chars // 2, width // 3))

    def tick(self) -> None:
        if self.running:
            self.query_one(WorkingLine).tick(self.stats.timer.elapsed(), self.stats.timer.calls, self.stats.tokens)

    def phase(self, label: str, detail: str = "") -> None:
        if label == PHASE_CANCELLING or not self.cancel_flag.is_set():
            self.query_one(WorkingLine).start(label, detail)

    def action_toggle_verbose(self) -> None:
        self.verbose = not self.verbose
        for card in self.query(ToolCard):
            card.render_body()
        self.note("showing full tool output; ctrl+o folds it again" if self.verbose else "tool output folded; click a card or ctrl+o to unfold")

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
        if self.rt.hooks and self.rt.session_id and self.rt.session_id != sid:
            self.rt.hooks.dispatch("sessionEnd", {"session": self.rt.session_id, "reason": SESSION_END_CLEAR}, SESSION_END_CLEAR)
        self.session_id = self.rt.session_id = sid
        self.clear_transcript()
        self.load_history()
        self.stats = RunStats(window=self.rt.context_window)
        self.refresh_status()

    def show_tool_result(self, name: str, args: dict[str, Any]) -> None:
        card = ToolCard("local", name, args)
        self.add(card)
        res = self.rt.registry.run(name, args, ToolContext(workspace=self.rt.workspace, session_id=self.session_id, extra_dirs=self.rt.extra_dirs))
        card.finish(res.ok, res.output, {}, res.artifact.ref if res.artifact else "")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt":
            return
        palette = self.query_one("#palette", OptionList)
        text = event.value
        if text.startswith("/") and " " not in text:
            palette.clear_options()
            for command in matching(text, self.user_commands)[: LIMITS.command_matches_shown]:
                help_text = command.help if len(command.help) <= LIMITS.palette_help_chars else command.help[: LIMITS.palette_help_chars - 1].rstrip() + self.glyphs.ellipsis
                palette.add_option(f"{command.usage:<{LIMITS.palette_usage_width}} {help_text}")
            palette.highlighted = None
            palette.remove_class("hidden") if palette.option_count else palette.add_class("hidden")
        else:
            palette.add_class("hidden")

    def palette_open(self) -> bool:
        return not self.query_one("#palette", OptionList).has_class("hidden")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if len(self.screen_stack) > 1:
            return action not in ("history", "palette_move", "palette_complete")
        if action == "history":
            return True
        return self.palette_open() if action.startswith("palette_") else True

    def action_cycle_mode(self) -> None:
        self.rt.mode = next_mode(self.rt.mode)
        self.refresh_status()
        self.note(f"mode {self.rt.mode.value}")

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
        if text in EXIT_WORDS:
            text = "/exit"
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
            dispatch(self, text, self.user_commands)

    def run_user_command(self, command: UserCommand, arg: str) -> None:
        if self.running:
            self.note("a run is in progress; esc cancels it", error=True)
            return
        if self.rt.provider is None:
            self.open_setup()
            return
        if command.agent:
            self.use_agent(command.agent)
        if command.model:
            self.switch_model(command.model)
        typed = f"/{command.name} {arg}".strip()
        self.add(Static(f"{self.glyphs.prompt} {typed}", classes="user"))
        self.begin_run(expand(command.template, arg), typed)

    def set_effort(self, value: str) -> None:
        value = value.lower()
        if value not in (EFFORT_OFF, *EFFORTS):
            self.note(f"usage: /effort {'|'.join(EFFORTS)}|{EFFORT_OFF}", error=True)
            return
        apply_effort(self.rt, "" if value == EFFORT_OFF else value)
        self.refresh_status()
        self.note(f"effort {value}")

    def retry(self) -> None:
        if self.running:
            self.note("a run is in progress; esc cancels it", error=True)
        elif not self.history:
            self.note("no earlier prompt to retry", error=True)
        elif self.rt.provider is None:
            self.open_setup()
        else:
            text = self.history[-1]
            self.add(Static(f"{self.glyphs.prompt} {text}", classes="user"))
            self.begin_run(text)

    def rename(self, title: str) -> None:
        if not title:
            self.note("usage: /rename <title>", error=True)
            return
        self.rt.store.rename(self.session_id, title)
        self._title = title
        self.refresh_status()
        self.note(f"renamed to {title!r}")

    def export(self) -> None:
        path = self.rt.workspace / TRANSCRIPT_TEMPLATE.format(sid=self.session_id)
        lines = [f"# superclaw transcript {self.session_id}", ""]
        for message in self.rt.store.replay(self.session_id):
            if not (message.content or message.tool_calls):
                continue
            lines += [f"## {message.role}", "", message.content or ""]
            lines += [f"- tool `{c.name}` {c.arguments}" for c in message.tool_calls]
            lines.append("")
        path.write_text("\n".join(lines))
        self.note(f"transcript written to {path}")

    def do_compact(self) -> None:
        if self.running:
            self.note("a run is in progress; esc cancels it", error=True)
        elif self.rt.provider is None:
            self.open_setup()
        else:
            self.running = True
            self.query_one("#hints").add_class("hidden")
            self.query_one(WorkingLine).start(PHASE_COMPACTING)
            self.compact_worker()

    @work(thread=True, exclusive=True)
    def compact_worker(self) -> None:
        pairs = self.rt.store.timeline(self.session_id)
        notes = self.rt.hooks.dispatch("preCompact", {"session": self.session_id, "trigger": "manual", "custom_instructions": ""}, "manual").context if self.rt.hooks else []

        def summarize(brief: str) -> str:
            return self.rt.provider.complete([Message(role="system", content=summary_instructions(notes)), Message(role="user", content=brief)], []).text

        try:
            result = compact_context([m for _, m in pairs], summarize=summarize, footer=TRANSCRIPT_NOTE.format(sid=self.session_id))
        except Exception as e:
            self.call_from_thread(self.finish_compact, f"compact failed: {clip(str(e), LIMITS.preview_error_chars)}", True)
            return
        if not result.compacted:
            self.call_from_thread(self.finish_compact, "context is already small", False)
            return
        self.rt.store.append(self.session_id, "compaction", {"summary": result.summary, "through_seq": pairs[result.removed - 1][0]})
        self.call_from_thread(self.finish_compact, f"compacted {result.removed} messages into a summary", False)

    def finish_compact(self, text: str, error: bool) -> None:
        self.running = False
        self.query_one(WorkingLine).stop()
        self.query_one("#hints").remove_class("hidden")
        self.note(text, error=error)

    def attach(self, raw: str, quiet: bool = False) -> Clip | None:
        if not raw:
            self.note("usage: /attach <path>", error=True)
            return None
        found = read_attachments([raw], (self.rt.workspace, *self.rt.extra_dirs, self.rt.settings.clipboard_dir))
        for problem in found.problems:
            self.note(f"attachment {problem}", error=True)
        if not found.text:
            return None
        clip = self.clips.add("Image" if found.images else "File", text=found.text, image=found.images[0] if found.images else "")
        self.query_one("#prompt", Input).insert_text_at_cursor(clip.marker + " ")
        if not quiet:
            self.note(f"{clip.marker} is {raw}; it goes with the next message that still contains the marker")
        return clip

    def take_paste(self, text: str) -> bool:
        dropped = clipboard.parse_drop(text)
        if dropped:
            markers = [clip.marker for path in dropped if (clip := self.attach(str(path), quiet=True))]
            self.note(f"attached {', '.join(markers)}; delete a marker to leave that file out")
            return bool(markers)
        lines = text.count("\n") + 1
        if lines >= LIMITS.paste_lines_threshold or len(text) > LIMITS.paste_chars_threshold:
            clip = self.clips.add_pasted(text, lines)
            self.query_one("#prompt", Input).insert_text_at_cursor(clip.marker + " ")
            return True
        return False

    def paste_clipboard(self) -> None:
        prompt = self.query_one("#prompt", Input)
        image = clipboard.image_bytes()
        if image is not None:
            data, mime = image
            folder = self.rt.settings.clipboard_dir
            folder.mkdir(parents=True, exist_ok=True)
            for stale in sorted(folder.glob("*.*"), key=lambda p: p.stat().st_mtime)[: -LIMITS.clipboard_keep]:
                stale.unlink(missing_ok=True)
            target = folder / f"clip-{int(time.time() * 1000)}.{mime.split('/')[1]}"
            target.write_bytes(data)
            clip = self.attach(str(target), quiet=True)
            if clip:
                self.note(f"{clip.marker} is the clipboard image ({len(data):,} bytes); it goes with the next message that still contains the marker")
            return
        pasted = clipboard.text()
        if pasted and not self.take_paste(pasted):
            prompt.insert_text_at_cursor(pasted.splitlines()[0])

    def use_agent(self, name: str) -> None:
        profiles = load_agents(self.rt.settings.agent_roots(self.rt.workspace))
        if not name:
            for profile in profiles:
                mark = self.glyphs.prompt if self.rt.agent and profile.name == self.rt.agent.name else " "
                self.note(f"{mark} {profile.name}: {profile.description}")
            self.note("usage: /agent <name>|none" + ("" if profiles else f"; no profiles in {self.rt.settings.agent_roots()[0]}"))
            return
        if name == "none":
            self.rt.agent = None
            self.rt.policy.scope_to(frozenset())
            self.note("agent cleared")
            return
        found = next((p for p in profiles if p.name == name), None)
        if found is None:
            self.note(f"unknown agent {name!r}; available: {', '.join(p.name for p in profiles) or 'none'}", error=True)
            return
        self.rt.agent = found
        self.rt.policy.scope_to(found.tools)
        self.note(f"agent {found.name} {self.glyphs.dot} {', '.join(sorted(found.tools)) or 'all tools'}")

    def begin_run(self, text: str, typed: str = "") -> None:
        self.remember(typed or text)
        picked = self.clips.select(text)
        self.clips.clear()
        text = picked.prompt
        self.running = True
        self.cancel_flag.clear()
        self.stats.timer.start()
        self.query_one("#hints").add_class("hidden")
        self.query_one(WorkingLine).start(PHASE_RECALLING)
        self.query_one("#prompt", Input).placeholder = "esc to cancel"
        self.run_prompt(text, picked.images)

    def action_cancel(self) -> None:
        palette = self.query_one("#palette", OptionList)
        if not palette.has_class("hidden"):
            palette.add_class("hidden")
            self.query_one("#prompt", Input).value = ""
        elif self.running:
            self.cancel_flag.set()
            self.phase(PHASE_CANCELLING)

    def action_interrupt(self) -> None:
        if self.running and not self.cancel_flag.is_set():
            self.action_cancel()
            self.note("cancelling; ctrl+c again to quit", error=True)
        else:
            self.exit()

    @work(thread=True, exclusive=True)
    def run_prompt(self, text: str, images: list[str]) -> None:
        callbacks = Callbacks(on_event=lambda event: self.call_from_thread(self.render_event, event),
                              on_permission=self.ask_permission, on_ask_user=self.ask_questions)
        try:
            result = run_once(self.rt, text, self.session_id, callbacks, cancelled=self.cancel_flag.is_set, images=images)
        except Exception as e:
            self.call_from_thread(self.finish, None, clip(str(e), LIMITS.preview_error_chars))
            return
        self.call_from_thread(self.finish, result)

    def render_event(self, event: dict[str, Any]) -> None:
        kind, child = event["type"], bool(event.get("child"))
        if kind == "text_delta" and not child:
            self.stream_text += event["text"]
            if self.streaming is None:
                self.streaming = Static("", markup=False)
                self.add(self.streaming)
                self.phase(PHASE_WRITING)
            self.streaming.update(self.stream_text)
        elif kind == "text" and not child:
            self.drop_stream()
            self.add(Markdown(event["text"]))
        elif kind in ("tool_call", "tool_result"):
            self.render_tool(event, child)
        elif kind == "context" and not child:
            self.phase(PHASE_THINKING, context_overview(event, self.glyphs))
        elif kind == "permission_request" and not child:
            self.phase(PHASE_WAITING, f"{event['tool']}  {target_of(event['tool'], event['args'])}")
        elif kind == "permission_decision" and not child:
            self.phase(*self.current_tool)
            self.note(describe(event, self.glyphs))
        elif kind == "usage" and not child:
            self.stats.used, self.stats.window = event["context_used"], event["context_window"]
            self.stats.tokens, self.stats.cost = event["run_total"], event["run_cost_usd"]
            self.stats.saved, self.stats.kept_out = event["saved_tokens"], event["kept_out_tokens"]
            self.refresh_status()
        elif line := describe(event, self.glyphs):
            self.note(line)

    def render_tool(self, event: dict[str, Any], child: bool) -> None:
        key = f"{event.get('child', '')}:{event['id']}"
        if event["type"] == "tool_call":
            card = ToolCard(event["id"], event["name"], event["args"], child=child)
            self.cards[key] = card
            self.add(card)
            self.stats.timer.calls += 1
            self.current_tool = (event["name"], card.target)
            self.phase(*self.current_tool)
            return
        card = self.cards.pop(key, None)
        if card:
            card.finish(event["ok"], event["output"], event.get("display") or {}, event.get("ref", ""))
        self.phase(PHASE_THINKING)

    def drop_stream(self) -> None:
        if self.streaming is not None:
            self.streaming.remove()
        self.streaming, self.stream_text = None, ""

    def finish(self, result: Result | None, error: str = "") -> None:
        self.running = False
        self.drop_stream()
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
