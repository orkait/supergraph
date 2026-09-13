from __future__ import annotations

import json
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from superclaw.runtime import clip
from superclaw.settings import LIMITS
from superclaw.tui.theme import ACCENT, ADD, ADD_ROW, DEL, DEL_ROW, MUTED

GLYPH_RUNNING = "◐"
GLYPH_OK = "✓"
GLYPH_FAILED = "✗"
TARGET_KEYS = ("path", "pattern", "command", "code", "name", "query", "ref", "task")
DIFF_TOOLS = {"edit_file", "write_file"}


def target_of(name: str, args: dict[str, Any]) -> str:
    for key in TARGET_KEYS:
        value = args.get(key)
        if value:
            return clip(str(value).splitlines()[0] if key in ("command", "code", "task") else str(value), LIMITS.card_arg_chars)
    return clip(json.dumps(args), LIMITS.card_arg_chars) if args else ""


def body_lines(name: str, output: str, display: dict[str, Any]) -> list[Text]:
    if name in DIFF_TOOLS and display.get("preview"):
        return [_diff_line(line) for line in display["preview"].splitlines() if not line.startswith(("---", "+++", "@@"))]
    return [Text(line, style=MUTED) for line in output.splitlines()]


def diff_counts(display: dict[str, Any]) -> tuple[int, int]:
    lines = [line for line in (display.get("preview") or "").splitlines() if not line.startswith(("---", "+++"))]
    return sum(line.startswith("+") for line in lines), sum(line.startswith("-") for line in lines)


def _diff_line(line: str) -> Text:
    if line.startswith("+"):
        return Text(line, style=f"{ADD} {ADD_ROW}")
    if line.startswith("-"):
        return Text(line, style=f"{DEL} {DEL_ROW}")
    return Text(line, style=MUTED)


class ToolCard(Vertical):
    def __init__(self, call_id: str, name: str, args: dict[str, Any], child: bool = False) -> None:
        super().__init__(classes="child" if child else "")
        self.call_id = call_id
        self.tool = name
        self.target = target_of(name, args)
        self.lines: list[Text] = []
        self.counts = (0, 0)
        self.expanded = False
        self.status = GLYPH_RUNNING
        self.add_class("running")

    def compose(self) -> ComposeResult:
        yield Static(self.head(), classes="head")
        yield Static("", classes="body")
        yield Static("", classes="more")

    def head(self) -> Text:
        text = Text(f"{self.status} ")
        text.append(self.tool, style=ACCENT)
        if self.target:
            text.append(f"  {self.target}", style=MUTED)
        if self.counts != (0, 0):
            text.append(f"  (+{self.counts[0]} ", style=ADD).append(f"-{self.counts[1]})", style=DEL)
        return text

    def finish(self, ok: bool, output: str, display: dict[str, Any], ref: str) -> None:
        self.remove_class("running")
        self.status = GLYPH_OK if ok else GLYPH_FAILED
        if not ok:
            self.add_class("failed")
        self.counts = diff_counts(display) if self.tool in DIFF_TOOLS else (0, 0)
        self.lines = body_lines(self.tool, output, display)
        if ref:
            self.lines.append(Text(f"§{ref}", style=MUTED))
        self.render_body()

    def render_body(self) -> None:
        self.query_one(".head", Static).update(self.head())
        shown = self.lines if self.expanded else self.lines[:LIMITS.card_body_lines]
        body = Text("\n").join(shown) if shown else Text("")
        self.query_one(".body", Static).update(body)
        hidden = len(self.lines) - len(shown)
        more = f"… {hidden} more lines, click to expand" if hidden > 0 else ("click to collapse" if self.expanded and len(self.lines) > LIMITS.card_body_lines else "")
        self.query_one(".more", Static).update(more)

    def on_click(self) -> None:
        if len(self.lines) > LIMITS.card_body_lines:
            self.expanded = not self.expanded
            self.render_body()
