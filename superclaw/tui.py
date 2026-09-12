from __future__ import annotations

import json
import threading
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Markdown, OptionList, Static

from superclaw.app import Runtime, run_once
from superclaw.loop import Result
from superclaw.policy import Mode

PROMPT_PLACEHOLDER = "Ask superclaw · /mode ask|auto|plan|unsafe · /new · /sessions · /quit"


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
            yield Static(args[:1200] + ("…" if len(args) > 1200 else ""), classes="args")
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
        self.dismiss(str(event.option.prompt))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_skip(self) -> None:
        self.dismiss("")


class SuperclawApp(App[None]):
    CSS = """
    #status { height: 1; background: $panel; color: $text-muted; padding: 0 1; }
    #transcript { padding: 0 1; }
    #transcript > .user { color: $accent; margin: 1 0 0 0; }
    #transcript > .note { color: $text-muted; }
    #transcript > .error { color: $error; }
    #transcript > Markdown { margin: 0 0 1 0; }
    #prompt { dock: bottom; }
    #dialog { width: 80%; max-width: 100; height: auto; padding: 1 2; background: $surface; border: thick $accent; }
    #dialog .title { text-style: bold; margin-bottom: 1; }
    #dialog .args { color: $text-muted; max-height: 12; overflow-y: auto; }
    #dialog .reason { margin: 1 0; }
    #dialog .buttons { height: auto; }
    #dialog Button { margin-right: 1; }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, rt: Runtime, session_id: str) -> None:
        super().__init__()
        self.rt = rt
        self.session_id = session_id
        self.tokens = 0

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        yield VerticalScroll(id="transcript")
        yield Input(placeholder=PROMPT_PLACEHOLDER, id="prompt")

    def on_mount(self) -> None:
        self.refresh_status()
        self.query_one("#prompt", Input).focus()

    def refresh_status(self) -> None:
        self.query_one("#status", Static).update(
            f"{self.rt.model}  ·  mode {self.rt.mode.value}  ·  session {self.session_id}  ·  {self.tokens} tokens"
        )

    def add(self, widget: Static | Markdown) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(widget)
        transcript.scroll_end(animate=False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self.command(text)
            return
        self.add(Static(f"> {text}", classes="user"))
        event.input.disabled = True
        self.run_prompt(text)

    def command(self, text: str) -> None:
        parts = text.split()
        if parts[0] in ("/quit", "/exit"):
            self.exit()
        elif parts[0] == "/mode" and len(parts) == 2 and parts[1] in Mode._value2member_map_:
            self.rt.mode = Mode(parts[1])
            self.refresh_status()
        elif parts[0] == "/new":
            self.session_id = self.rt.store.create(cwd=str(self.rt.workspace), model=self.rt.model)
            self.query_one("#transcript", VerticalScroll).remove_children()
            self.refresh_status()
        elif parts[0] == "/sessions":
            for s in self.rt.store.list()[:20]:
                self.add(Static(f"{s['id']}  {s['event_count']} events  {s['cwd']}", classes="note"))
        else:
            self.add(Static(f"unknown command: {text}", classes="error"))

    @work(thread=True, exclusive=True)
    def run_prompt(self, text: str) -> None:
        result = run_once(
            self.rt, text, self.session_id,
            on_event=lambda event: self.call_from_thread(self.render_event, event),
            on_permission=self.ask_permission,
            on_ask_user=self.ask_questions,
        )
        self.call_from_thread(self.finish, result)

    def render_event(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "text":
            self.add(Markdown(event["text"]))
        elif kind == "tool_call":
            args = json.dumps(event["args"])
            self.add(Static(f"→ {event['name']} {args[:160]}{'…' if len(args) > 160 else ''}", classes="note"))
        elif kind == "tool_result":
            if not event["ok"]:
                first = event["output"].splitlines()[0] if event["output"] else ""
                self.add(Static(f"✗ {event['name']}: {first[:200]}", classes="error"))
            for path in event.get("changed_files", []):
                self.add(Static(f"✎ {path}", classes="note"))
        elif kind == "permission_decision":
            self.add(Static(f"permission {event['tool']}: {event['decision']}", classes="note"))
        elif kind == "compaction":
            self.add(Static(f"compacted {event['removed']} messages", classes="note"))
        elif kind == "usage":
            self.tokens += event["input_tokens"] + event["output_tokens"]
            self.refresh_status()

    def finish(self, result: Result) -> None:
        if result.stop_reason or result.incomplete:
            self.add(Static(f"stopped: {result.stop_reason or result.incomplete_reason}", classes="error"))
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()

    def modal(self, screen: ModalScreen[str]) -> str:
        done = threading.Event()
        answer: dict[str, str] = {}

        def settle(value: str | None) -> None:
            answer["value"] = value or ""
            done.set()

        self.call_from_thread(self.push_screen, screen, settle)
        done.wait()
        return answer["value"]

    def ask_permission(self, request: dict[str, Any]) -> str:
        return self.modal(PermissionScreen(request)) or "deny"

    def ask_questions(self, questions: list[dict[str, Any]]) -> list[str]:
        return [self.modal(QuestionScreen(q)) for q in questions]
