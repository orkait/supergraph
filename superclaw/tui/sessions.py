from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from superclaw.dsl import age
from superclaw.runtime import compact
from superclaw.settings import LIMITS
from superclaw.tui.theme import ACCENT, MUTED

UNTITLED = "(untitled)"
NS_PER_MS = 1_000_000


def project_of(cwd: str) -> str:
    return Path(cwd).name or cwd


def size_of(session: dict[str, Any]) -> str:
    stored = int(session.get("bytes") or 0)
    return f"{compact(stored)}B" if stored else f"{session.get('event_count', 0)} events"


class ResumeScreen(ModalScreen[str]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+c", "app.interrupt", "Quit"),
        ("down", "move(1)", "Next"),
        ("up", "move(-1)", "Previous"),
        ("ctrl+a", "everywhere", "All projects"),
    ]

    def __init__(self, sessions: list[dict[str, Any]], workspace: Path, active: str) -> None:
        super().__init__()
        self.sessions = sessions
        self.here = str(workspace)
        self.active = active
        self.everywhere = False
        self.rows: list[dict[str, Any] | None] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Resume session", classes="title")
            yield Input(placeholder="type to search", id="filter")
            yield OptionList(id="sessions")
            yield Static("", id="hint", classes="reason")

    def on_mount(self) -> None:
        self.refill()
        self.query_one("#filter", Input).focus()

    def matches(self, session: dict[str, Any], text: str) -> bool:
        haystack = " ".join(str(session.get(key, "")) for key in ("title", "cwd", "branch", "id", "model")).lower()
        return all(word in haystack for word in text.split())

    def groups(self, text: str) -> list[tuple[str, list[dict[str, Any]]]]:
        shown = [s for s in self.sessions if self.matches(s, text)]
        mine = [s for s in shown if s.get("cwd") == self.here]
        if not self.everywhere:
            return [(project_of(self.here), mine)]
        others: dict[str, list[dict[str, Any]]] = {}
        for session in shown:
            if session.get("cwd") != self.here:
                others.setdefault(project_of(str(session.get("cwd", ""))), []).append(session)
        return [(project_of(self.here), mine), *sorted(others.items())]

    def refill(self) -> None:
        text = self.query_one("#filter", Input).value.strip().lower()
        options: list[Option] = []
        self.rows = []
        for project, sessions in self.groups(text):
            if not sessions:
                continue
            options.append(Option(Text(project, style=MUTED), disabled=True))
            self.rows.append(None)
            for session in sessions[: LIMITS.resume_per_project]:
                options.append(Option(self.row(session)))
                self.rows.append(session)
        table = self.query_one("#sessions", OptionList)
        table.clear_options()
        table.add_options(options or [Option(Text("no session matches", style=MUTED), disabled=True)])
        self.rows = self.rows or [None]
        table.highlighted = next((i for i, s in enumerate(self.rows) if s is not None), None)
        dot = self.app.glyphs.dot
        scope = "this project" if not self.everywhere else "all projects"
        self.query_one("#hint", Static).update(f" {dot} ".join(("enter resumes", "ctrl+a " + ("all projects" if not self.everywhere else "this project"), f"showing {scope}", "esc cancels")))

    def row(self, session: dict[str, Any]) -> Text:
        dot = self.app.glyphs.dot
        title = str(session.get("title") or "").strip() or UNTITLED
        head = Text(title, style=ACCENT if session["id"] == self.active else "")
        facts = [age(int(session.get("created", 0)) // NS_PER_MS)]
        if session.get("branch"):
            facts.append(str(session["branch"]))
        facts += [size_of(session), str(session.get("model", ""))]
        return head + Text(f"\n{f' {dot} '.join(f for f in facts if f)}", style=MUTED)

    def picked(self) -> str:
        index = self.query_one("#sessions", OptionList).highlighted
        row = self.rows[index] if index is not None and index < len(self.rows) else None
        return str(row["id"]) if row else ""

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        self.refill()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if chosen := self.picked():
            self.dismiss(chosen)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        row = self.rows[event.option_index]
        if row is not None:
            self.dismiss(str(row["id"]))

    def action_move(self, step: int) -> None:
        table = self.query_one("#sessions", OptionList)
        table.action_cursor_down() if step > 0 else table.action_cursor_up()

    def action_everywhere(self) -> None:
        self.everywhere = not self.everywhere
        self.refill()

    def action_cancel(self) -> None:
        self.dismiss("")
