from __future__ import annotations

import json
import time
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from superclaw.runtime import clip, count
from superclaw.settings import LIMITS
from superclaw.tui.theme import ACCENT, ADD, ADD_ROW, DEL, DEL_ROW, MUTED

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
        self.ok: bool | None = None
        self.started = time.monotonic()
        self.elapsed = 0.0
        self.produced = 0
        self.add_class("running")

    def compose(self) -> ComposeResult:
        yield Static(self.head(), classes="head")
        yield Static("", classes="body")
        yield Static("", classes="more")

    def head(self) -> Text:
        glyphs = self.app.glyphs
        status = glyphs.running if self.ok is None else (glyphs.ok if self.ok else glyphs.failed)
        text = Text(f"{status} ")
        text.append(self.tool, style=ACCENT)
        if self.target:
            text.append(f"  {self.target}", style=MUTED)
        if self.counts != (0, 0):
            text.append(f"  (+{self.counts[0]} ", style=ADD).append(f"-{self.counts[1]})", style=DEL)
        elif self.ok is not None:
            text.append(f"  {glyphs.dot} {count(self.produced, 'line')}", style=MUTED)
        if self.ok is not None:
            text.append(f"  {self.elapsed:.1f}s", style=MUTED)
        return text

    def finish(self, ok: bool, output: str, display: dict[str, Any], ref: str) -> None:
        self.remove_class("running")
        self.ok = ok
        self.elapsed = time.monotonic() - self.started
        if not ok:
            self.add_class("failed")
        self.counts = diff_counts(display) if self.tool in DIFF_TOOLS else (0, 0)
        self.lines = body_lines(self.tool, output, display)
        self.produced = len(self.lines)
        if ref:
            self.lines.append(Text(f"§{ref}", style=MUTED))
        if self.children:
            self.render_body()

    def on_mount(self) -> None:
        self.render_body()

    def folded(self) -> int:
        return LIMITS.card_body_lines if self.tool in DIFF_TOOLS or self.ok is False else 0

    def render_body(self) -> None:
        self.query_one(".head", Static).update(self.head())
        unfolded = self.expanded or self.app.verbose  # type: ignore[attr-defined]
        shown = self.lines if unfolded else self.lines[: self.folded()]
        body = self.query_one(".body", Static)
        body.update(Text("\n").join(shown) if shown else Text(""))
        body.display = bool(shown)
        hidden = len(self.lines) - len(shown)
        if hidden > 0:
            more = f"{self.app.glyphs.ellipsis} {count(hidden, 'more line') if shown else count(hidden, 'line')}, click or ctrl+o to expand"
        else:
            more = "click to collapse" if unfolded and len(self.lines) > self.folded() else ""
        hint = self.query_one(".more", Static)
        hint.update(more)
        hint.display = bool(more)

    def on_click(self) -> None:
        if len(self.lines) > self.folded():
            self.expanded = not self.expanded
            self.render_body()
